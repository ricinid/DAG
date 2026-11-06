#include "dag_runtime.h"
#include <zlib.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr size_t kMaxZipBytes = 32u * 1024u * 1024u;
constexpr size_t kMaxInflatedBytes = 64u * 1024u * 1024u;
using DagClock = std::chrono::steady_clock;
uint64_t elapsed_ns(DagClock::time_point a, DagClock::time_point b) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}

uint16_t u16(const uint8_t * p) { return uint16_t(p[0]) | uint16_t(p[1]) << 8; }
uint32_t u32(const uint8_t * p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 |
                                            uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
uint64_t u64(const uint8_t * p) { return uint64_t(u32(p)) | uint64_t(u32(p + 4)) << 32; }
uint32_t required(const std::string & s, const char * key) {
    auto pos = s.find(std::string("\"") + key + "\"");
    if (pos == std::string::npos) throw std::runtime_error(std::string("Missing field ") + key);
    pos = s.find(':', pos);
    if (pos == std::string::npos) throw std::runtime_error("Malformed JSON");
    pos = s.find_first_of("0123456789", pos + 1);
    if (pos == std::string::npos) throw std::runtime_error("Malformed JSON numeric field");
    auto n = std::strtoull(s.c_str() + pos, nullptr, 10);
    if (n > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("Field overflow");
    return static_cast<uint32_t>(n);
}
uint64_t required64(const std::string & s, const char * key) {
    auto pos = s.find(std::string("\"") + key + "\"");
    if (pos == std::string::npos) throw std::runtime_error(std::string("Missing field ") + key);
    pos = s.find(':', pos);
    if (pos == std::string::npos) throw std::runtime_error("Malformed JSON");
    pos = s.find_first_of("0123456789", pos + 1);
    if (pos == std::string::npos) throw std::runtime_error("Malformed JSON numeric field");
    return std::strtoull(s.c_str() + pos, nullptr, 10);
}
std::string string_field(const std::string & s, const char * key) {
    auto pos = s.find(std::string("\"") + key + "\"");
    if (pos == std::string::npos) throw std::runtime_error(std::string("Missing field ") + key);
    pos = s.find(':', pos);
    if (pos == std::string::npos) throw std::runtime_error("Malformed JSON");
    pos = s.find('"', pos + 1);
    if (pos == std::string::npos) throw std::runtime_error("Malformed JSON string");
    auto end = s.find('"', pos + 1);
    if (end == std::string::npos) throw std::runtime_error("Malformed JSON string");
    return s.substr(pos + 1, end - pos - 1);
}
std::vector<uint8_t> inflate_bytes(const uint8_t * src, size_t srcsz, int mode) {
    if (srcsz > kMaxZipBytes) throw std::runtime_error("Compressed unit exceeds safety limit");
    z_stream zs{};
    if (inflateInit2(&zs, mode) != Z_OK) throw std::runtime_error("zlib init failed");
    std::vector<uint8_t> output;
    try {
        size_t out_pos = 0;
        int ret = Z_OK;
        while (ret != Z_STREAM_END) {
            if (output.size() == out_pos) {
                if (output.size() >= kMaxInflatedBytes) throw std::runtime_error("Zip bomb limit");
                output.resize(std::min(kMaxInflatedBytes, std::max<size_t>(4096, output.size() * 2)));
            }
            if (zs.avail_in == 0 && zs.total_in < srcsz) {
                zs.next_in = const_cast<Bytef *>(reinterpret_cast<const Bytef *>(src + zs.total_in));
                zs.avail_in = static_cast<uInt>(std::min<size_t>(srcsz - zs.total_in, std::numeric_limits<uInt>::max()));
            }
            zs.next_out = output.data() + out_pos;
            zs.avail_out = static_cast<uInt>(output.size() - out_pos);
            ret = inflate(&zs, Z_NO_FLUSH);
            out_pos = zs.total_out;
            if (ret != Z_OK && ret != Z_STREAM_END) throw std::runtime_error("Corrupt zlib payload");
            if (zs.avail_in == 0 && zs.total_in >= srcsz && ret != Z_STREAM_END)
                throw std::runtime_error("Truncated compressed payload");
        }
        output.resize(out_pos);
    } catch (...) { inflateEnd(&zs); throw; }
    inflateEnd(&zs);
    return output;
}
struct Record {
    uint64_t offset = 0;
    uint32_t size = 0, row = 0, rows = 0, col = 0, cols = 0;
};
struct Tensor {
    std::vector<Record> records;
    uint64_t area = 0;
    uint32_t rows = 0, cols = 0;
};
struct Unit {
    uint32_t dtype = 0, rows = 0, cols = 0;
    std::vector<uint8_t> meta;
    std::vector<uint8_t> dags;
    size_t bytes() const { return meta.capacity() + dags.capacity() + sizeof(Unit); }
};
std::vector<uint8_t> zip_field(const uint8_t * p, size_t n, const std::string & want) {
    size_t k = 0;
    while (k + 30 <= n && u32(p + k) == 0x04034b50) {
        uint16_t flag = u16(p + k + 6), method = u16(p + k + 8);
        if (flag & 0x0008) throw std::runtime_error("ZIP data descriptors unsupported");
        uint32_t packed = u32(p + k + 18), length = u32(p + k + 22);
        size_t fname = u16(p + k + 26), extra = u16(p + k + 28);
        size_t data = k + 30 + fname + extra;
        if (data > n || packed > n - data) throw std::runtime_error("Malformed ZIP boundaries");
        std::string name(reinterpret_cast<const char *>(p + k + 30), fname);
        if (name == want) {
            if (method == 0) return std::vector<uint8_t>(p + data, p + data + packed);
            if (method == 8) {
                auto ret = inflate_bytes(p + data, packed, -MAX_WBITS);
                if (ret.size() != length) throw std::runtime_error("ZIP uncompressed size mismatch");
                return ret;
            }
            throw std::runtime_error("Unsupported ZIP compression method");
        }
        k = data + packed;
    }
    throw std::runtime_error("ZIP entry not found: " + want);
}
float half_to_float(uint16_t half) {
    uint32_t s = (half & 0x8000u) << 16;
    uint32_t e = (half >> 10) & 31u, f = half & 1023u;
    uint32_t bits;
    if (!e) {
        if (!f) bits = s;
        else {
            int exp = -14;
            while (!(f & 1024u)) { f <<= 1; --exp; }
            f &= 1023u;
            bits = s | (uint32_t(exp + 127) << 23) | (f << 13);
        }
    } else if (e == 31u) bits = s | 0x7f800000u | (f << 13);
    else bits = s | ((e + 112u) << 23) | (f << 13);
    float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
void scale_offset(const uint8_t * hdr, uint32_t dtype, uint32_t g, double & scale, double & offset) {
    if (dtype == 2) {
        scale = half_to_float(u16(hdr)); offset = 0.; return;
    }
    if (dtype != 12 || g >= 8) throw std::runtime_error("Unknown dtype");
    const auto * c = hdr + 4;
    uint32_t sc, mn;
    if (g < 4) { sc = c[g] & 63u; mn = c[g+4] & 63u; }
    else { sc = (c[g+4] & 15u) | ((c[g-4] >> 6) << 4);
           mn = (c[g+4] >> 4) | ((c[g] >> 6) << 4); }
    scale = double(half_to_float(u16(hdr))) * sc;
    offset = 8.0 * scale - double(half_to_float(u16(hdr + 2))) * mn;
}
struct Cursor {
    const uint8_t * p, * end;
    explicit Cursor(const std::vector<uint8_t> & v):p(v.data()),end(v.data()+v.size()){}
    uint32_t get32() {
        if (size_t(end - p) < 4) throw std::runtime_error("Truncated DAG IR");
        uint32_t x = u32(p); p+=4; return x;
    }
    int32_t geti32() { return static_cast<int32_t>(get32()); }
    void checksize(size_t n) const { if (size_t(end-p)<n) throw std::runtime_error("Truncated DAG record"); }
};
class Engine {
public:
    explicit Engine(const std::string & dir, uint64_t budget):budget_(budget) {
        // Production snapshots are intentionally sparse. Targeted A/B smoke tests
        // request every GEMV because Windows TerminateProcess bypasses atexit.
        if (const char * setting = std::getenv("DAG_STATS_EVERY")) {
            char * end = nullptr;
            errno = 0;
            auto parsed = std::strtoull(setting, &end, 10);
            if (!*setting || end == setting || *end != '\0' || errno == ERANGE || parsed == 0)
                throw std::runtime_error("DAG_STATS_EVERY must be a positive integer");
            stats_every_ = static_cast<uint64_t>(parsed);
        }
        auto path = dir + "/weights.index.jsonl";
        std::ifstream index(path);
        if (!index) throw std::runtime_error("Cannot open " + path);
        std::string line;
        uint64_t next_seq = 0;
        while (std::getline(index, line)) {
            if (line.empty()) continue;
            auto seq = required64(line, "sequence");
            if (seq != next_seq++) throw std::runtime_error("Index sequence not contiguous");
            Record r{};
            r.offset=required64(line,"offset");r.size=required(line,"size");
            r.row=required(line,"row_start"); r.rows=required(line,"rows");
            r.col=required(line,"col_start"); r.cols=required(line,"cols");
            auto & t = tensors_[string_field(line,"tensor")];
            t.records.push_back(r);
            t.rows=std::max(t.rows,r.row+r.rows);t.cols=std::max(t.cols,r.col+r.cols);
            t.area += uint64_t(r.rows)*r.cols;
        }
        file_ = std::fopen((dir+"/weights.adagstream").c_str(),"rb");
        if (!file_) throw std::runtime_error("Cannot open weights.adagstream");
        char magic[11]{};
        if (std::fread(magic,1,11,file_)!=11 || std::memcmp(magic,"ADAGSTREAM\x01",11)!=0)
            throw std::runtime_error("Invalid ADAGSTREAM magic");
        std::fprintf(stderr,"[DAG native] tensors=%zu records=%llu cache_limit=%llu bytes\n",
                     tensors_.size(), (unsigned long long)next_seq, (unsigned long long)budget_);
    }
    ~Engine() { if (file_) std::fclose(file_); }
    int gemv(const char * tensor_name, const float * x, uint64_t n, float * y, uint64_t m) {
        if (!tensor_name || !x || !y) return -1;
        auto it = tensors_.find(tensor_name);
        if (it == tensors_.end()) return 0;
        const Tensor & t = it->second;
        if (t.rows != m || t.cols != n || t.area != m*n) {
            std::fprintf(stderr,"[DAG native] shape or coverage mismatch: %s\n",tensor_name);return -1;
        }
        try {
            const auto gemv_start = DagClock::now();
            const uint64_t misses_before = misses_;
            uint64_t local_load_ns = 0, local_execute_ns = 0;
            std::vector<double> accum(m, 0.);
            for (const auto & r : t.records) {
                const auto load_start = DagClock::now();
                auto u = load(r);
                const auto execute_start = DagClock::now();
                execute(*u,r,x,accum);
                const auto execute_end = DagClock::now();
                local_load_ns += elapsed_ns(load_start, execute_start);
                local_execute_ns += elapsed_ns(execute_start, execute_end);
            }
            for (size_t i=0;i<m;i++) y[i]=static_cast<float>(accum[i]);
            // Exclude diagnostic logging from the measured kernel duration.
            gemv_last_ns_ = elapsed_ns(gemv_start, DagClock::now());
            gemv_last_load_ns_ = local_load_ns;
            gemv_last_execute_ns_ = local_execute_ns;
            if (matvecs_ == 0) gemv_first_ns_ = gemv_last_ns_;
            gemv_total_ns_ += gemv_last_ns_;
            gemv_load_total_ns_ += local_load_ns;
            gemv_execute_total_ns_ += local_execute_ns;
            if (misses_before == misses_) {
                ++gemv_warm_calls_;
                gemv_warm_total_ns_ += gemv_last_ns_;
            } else {
                ++gemv_cold_calls_;
            }
            ++matvecs_;
            if (matvecs_ == 1 || matvecs_ % stats_every_ == 0) {
                std::fprintf(stderr,"[DAG native] STATS %s\n",stats().c_str());
                std::fflush(stderr);
            }
            return 1;
        } catch (const std::exception & e) {
            std::fprintf(stderr,"[DAG native] failed at %s: %s\n",tensor_name,e.what()); return -1;
        }
    }
    std::string stats() const {
        std::ostringstream out;
        out << "{\"stats_every\":" << stats_every_ << ",\"gemv_calls\":" << matvecs_ << ",\"cache_hits\":" << hits_ <<
               ",\"cache_misses\":" << misses_ << ",\"stream_bytes_read\":" << read_bytes_ <<
               ",\"cache_payload_bytes\":" << cache_used_ << ",\"cache_budget_bytes\":" << budget_ <<
               ",\"gemv_first_ns\":" << gemv_first_ns_ <<
               ",\"gemv_last_ns\":" << gemv_last_ns_ <<
               ",\"gemv_last_load_ns\":" << gemv_last_load_ns_ <<
               ",\"gemv_last_execute_ns\":" << gemv_last_execute_ns_ <<
               ",\"gemv_total_ns\":" << gemv_total_ns_ <<
               ",\"gemv_load_total_ns\":" << gemv_load_total_ns_ <<
               ",\"gemv_execute_total_ns\":" << gemv_execute_total_ns_ <<
               ",\"gemv_cold_calls\":" << gemv_cold_calls_ <<
               ",\"gemv_warm_calls\":" << gemv_warm_calls_ <<
               ",\"gemv_warm_total_ns\":" << gemv_warm_total_ns_ <<
               ",\"stream_io_total_ns\":" << stream_io_total_ns_ <<
               ",\"unit_materialize_total_ns\":" << unit_materialize_total_ns_ << "}";
        return out.str();
    }
private:
    uint64_t budget_; uint64_t stats_every_ = 1000; std::FILE * file_ = nullptr;
    std::unordered_map<std::string, Tensor> tensors_;
    std::list<std::pair<uint64_t,std::shared_ptr<Unit>>> cache_;
    std::unordered_map<uint64_t,decltype(cache_)::iterator> lookup_;
    size_t cache_used_ = 0;
    uint64_t hits_=0, misses_=0, read_bytes_=0, matvecs_=0;
    // Monotonic wall-time accounting, sampled only for GEMVs completed successfully.
    // Timings use steady_clock, exclude STATS logging and include profiler overhead.
    uint64_t gemv_first_ns_=0, gemv_last_ns_=0, gemv_total_ns_=0;
    uint64_t gemv_last_load_ns_=0, gemv_last_execute_ns_=0;
    uint64_t gemv_load_total_ns_=0, gemv_execute_total_ns_=0;
    uint64_t gemv_cold_calls_=0, gemv_warm_calls_=0, gemv_warm_total_ns_=0;
    uint64_t stream_io_total_ns_=0, unit_materialize_total_ns_=0;
    std::shared_ptr<Unit> load(const Record & r) {
        auto it = lookup_.find(r.offset);
        if (it != lookup_.end()) {
            cache_.splice(cache_.begin(),cache_,it->second);
            ++hits_; return cache_.front().second;
        }
        ++misses_;
        const auto read_start = DagClock::now();
#if defined(_WIN32)
        if (_fseeki64(file_, static_cast<int64_t>(r.offset), SEEK_SET)) throw std::runtime_error("Seek failed");
#else
        if (fseeko(file_, static_cast<off_t>(r.offset), SEEK_SET)) throw std::runtime_error("Seek failed");
#endif
        uint8_t nbytes[8];
        if (std::fread(nbytes,1,8,file_) != 8 || u64(nbytes) != r.size)
            throw std::runtime_error("Record size mismatch");
        if (r.size > kMaxZipBytes) throw std::runtime_error("Record exceeds safety limit");
        std::vector<uint8_t> zip(r.size);
        if (std::fread(zip.data(),1,r.size,file_) != r.size) throw std::runtime_error("Truncated record");
        const auto read_end = DagClock::now();
        stream_io_total_ns_ += elapsed_ns(read_start, read_end);
        read_bytes_ += r.size + 8;
        auto u=std::make_shared<Unit>();
        auto info=zip_field(zip.data(),zip.size(),"info.json");
        std::string json(info.begin(),info.end());
        u->dtype=required(json,"dtype");u->rows=required(json,"rows");u->cols=required(json,"cols");
        if (u->dtype != 2 && u->dtype != 12) throw std::runtime_error("Unsupported dtype");
        if (u->rows!=r.rows || u->cols!=r.cols || required(json,"row_start")!=r.row ||
            required(json,"col_start")!=r.col) throw std::runtime_error("Unit/index mismatch");
        u->meta=zip_field(zip.data(),zip.size(),"meta.bin");
        auto deflated=zip_field(zip.data(),zip.size(),"dags.bin");
        u->dags=inflate_bytes(deflated.data(),deflated.size(),MAX_WBITS);
        if (u->dags.size()>kMaxInflatedBytes) throw std::runtime_error("Large DAG");
        auto extra = u->bytes();
        if ((!budget_ || extra<=budget_)) {
            while (budget_ && !cache_.empty() && cache_used_ + extra > budget_) {
                auto & tail=cache_.back();
                cache_used_ -= tail.second->bytes();
                lookup_.erase(tail.first);
                cache_.pop_back();
            }
            cache_.push_front({r.offset,u});
            lookup_[r.offset]=cache_.begin();
            cache_used_ += extra;
        }
        // ZIP decompression, metadata parsing, DAG inflation and cache management.
        unit_materialize_total_ns_ += elapsed_ns(read_end, DagClock::now());
        return u;
    }
    static void execute(const Unit & u, const Record & r, const float * x, std::vector<double> & y) {
        const uint32_t header_len = u.dtype == 2 ? 2 : 16;
        const uint32_t qblock = u.dtype == 2 ? 32 : 256;
        if (u.cols % qblock != 0) throw std::runtime_error("Unaligned unit width");
        const uint32_t blocks = u.cols/qblock, groups = u.cols/32;
        if (u.meta.size()!=size_t(u.rows)*blocks*header_len)
            throw std::runtime_error("Metadata size mismatch");
        Cursor cur(u.dags);
        if (cur.get32()!=groups) throw std::runtime_error("Wrong DAG group count");
        std::vector<double> values;
        for (uint32_t g=0;g<groups;g++) {
            auto length=cur.get32();cur.checksize(length);
            const uint8_t * group_end=cur.p+length;
            uint32_t inputs=cur.get32(), outputs=cur.get32(), nodes=cur.get32();
            if (inputs!=32 || outputs!=u.rows || nodes>4096)
                throw std::runtime_error("Invalid DAG header");
            values.resize(inputs+nodes);
            double sx=0.;
            for (uint32_t i=0;i<inputs;i++) {
                double v=x[r.col+g*32+i]; values[i]=v; sx+=v;
            }
            for (uint32_t j=0;j<nodes;j++) {
                auto a=cur.get32(),b=cur.get32();
                int32_t ca=cur.geti32(),cb=cur.geti32();
                if (a>=inputs+j||b>=inputs+j) throw std::runtime_error("Non-topological DAG");
                values[inputs+j]=double(ca)*values[a]+double(cb)*values[b];
            }
            for (uint32_t rr=0;rr<u.rows;rr++) {
                uint32_t nt=cur.get32();
                if (nt>10240) throw std::runtime_error("Output term count unreasonable");
                double terms=0.;
                for (uint32_t term=0;term<nt;term++) {
                    auto id=cur.get32();int32_t c=cur.geti32();
                    if (id>=inputs+nodes) throw std::runtime_error("Invalid output reference");
                    terms+=double(c)*values[id];
                }
                uint32_t b=g/(qblock/32), local=g%(qblock/32);
                const uint8_t *hdr=&u.meta[(rr*blocks+b)*header_len];
                double scale,offset;
                scale_offset(hdr,u.dtype,local,scale,offset);
                y[r.row+rr] += scale*terms+offset*sx;
            }
            if (cur.p != group_end) throw std::runtime_error("DAG group length mismatch");
        }
        if (cur.p!=cur.end) throw std::runtime_error("Trailing DAG bytes");
    }
};

std::once_flag singleton_flag;
std::unique_ptr<Engine> singleton;
void init_singleton(){
    const char * dir = std::getenv("DAG_STREAM_DIR");
    if (!dir || !*dir) return;
    uint64_t budget=0;
    if (const char * b=std::getenv("DAG_CACHE_BYTES")) budget=std::strtoull(b,nullptr,10);
    try { singleton=std::make_unique<Engine>(dir,budget); }
    catch(const std::exception & e){std::fprintf(stderr,"[DAG native] init failed: %s\n",e.what());std::abort();}
    std::atexit([](){if(singleton)std::fprintf(stderr,"[DAG native] STATS %s\n",singleton->stats().c_str());});
}
} // namespace
extern "C" void * dag_context_open(const char * dir, uint64_t budget) {
    if(!dir) return nullptr;
    try { return new Engine(dir,budget); }
    catch(const std::exception & e) { std::fprintf(stderr,"[DAG native] open: %s\n",e.what());return nullptr; }
}
extern "C" void dag_context_close(void * ctx) { delete static_cast<Engine *>(ctx); }
extern "C" int dag_context_gemv(void * ctx, const char * name, const float * x, uint64_t n,
                                  float * y, uint64_t m) {
    if (!ctx) return -1;
    return static_cast<Engine *>(ctx)->gemv(name,x,n,y,m);
}
extern "C" int dag_context_stats(void * ctx, char * dst, size_t capacity) {
    if (!ctx||!dst||!capacity) return -1;
    auto s=static_cast<Engine *>(ctx)->stats();
    if (s.size()+1>capacity) return -1;
    std::memcpy(dst,s.c_str(),s.size()+1);return static_cast<int>(s.size());
}
extern "C" int dag_try_ggml_gemv(const char * name, int ith, const float * x,
                                   uint64_t n, float * y, uint64_t m) {
    // Exact-name optional scope control for safe single-tensor integration tests.
    if (const char * allow = std::getenv("DAG_TENSOR_ALLOW")) {
        if (std::strcmp(allow,"all") != 0 && std::strcmp(allow,"*") != 0) {
            bool found = false;
            const char * cur = allow;
            while (*cur) {
                const char * sep = std::strchr(cur,',');
                const size_t len = sep ? size_t(sep-cur) : std::strlen(cur);
                if (name && std::strlen(name) == len && std::memcmp(name,cur,len) == 0) {
                    found=true;break;
                }
                if (!sep) break;
                cur=sep+1;
            }
            if (!found) return 0;
        }
    }
    std::call_once(singleton_flag,init_singleton);
    if (!singleton) return 0;
    // The GGML hook MUST restrict routing to nth == 1. Multi-threaded matmul
    // falls through to the original GGML implementation.
    if (ith != 0) return 0; // GGML integration only enables this hook when nth == 1.
    int rc=singleton->gemv(name,x,n,y,m);
    if (rc<0) std::abort();
    return rc;
}