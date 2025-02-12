/*
 * Copyright (c) 2022, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *	this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 *those of the authors and should not be interpreted as representing official
 *policies, either expressed or implied, of the FreeBSD Project.
 */

// Matrices are SIZE*SIZE..  POT should be efficiently implemented in CUBLAS
#include <ios>
#define SIZE 2048ul
#define USEMEM 0.9 // Try to allocate 90% of memory

// Used to report op/s, measured through Visual Profiler, CUBLAS from CUDA 7.5
// (Seems that they indeed take the naive dim^3 approach)
// #define OPS_PER_MUL 17188257792ul // Measured for SIZE = 2048
#define OPS_PER_MUL 1100048498688ul // Extrapolated for SIZE = 8192

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <errno.h>
#include <exception>
#include <fstream>
#include <signal.h>
#include <stdexcept>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <vector>

#define SIGTERM_TIMEOUT_THRESHOLD_SECS                                         \
    30 // number of seconds for sigterm to kill child processes before forcing a
       // sigkill

#include "cublas_v2.h"
#define CUDA_ENABLE_DEPRECATEDC
#include <cuda.h>

void log(std::string msg) {
    char buffer[26];
    int millisec;
    struct tm *tm_info;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    millisec = lrint(tv.tv_usec / 1000.0);
    if (millisec >= 1000) {
        millisec -= 1000;
        tv.tv_sec++;
    }
    tm_info = localtime(&tv.tv_sec);

    std::ofstream outfile;
    strftime(buffer, 26, "%Y:%m:%d %H:%M:%S", tm_info);
    outfile.open("out.log", std::ios_base::app);
    outfile << std::string(buffer) << "." << millisec << " - " << msg << "\n";
    outfile.close();
    return;
}

volatile sig_atomic_t terminateFlag = 0;

void terminateSelf(int signum) { terminateFlag = 1; }

void _checkError(int rCode, std::string file, int line, std::string desc = "") {
    if (rCode != CUDA_SUCCESS) {
        const char *err;
        cuGetErrorString((CUresult)rCode, &err);

        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

void _checkError(cublasStatus_t rCode, std::string file, int line,
                 std::string desc = "") {
    if (rCode != CUBLAS_STATUS_SUCCESS) {
#if CUBLAS_VER_MAJOR >= 12
        const char *err = cublasGetStatusString(rCode);
#else
        const char *err = "";
#endif
        throw std::runtime_error(
            (desc == "" ? std::string("Error (")
                        : (std::string("Error in ") + desc + " (")) +
            file + ":" + std::to_string(line) + "): " + err);
        // Yes, this *is* a memory leak, but this block is only executed on
        // error, so it's not a big deal
    }
}

#define checkError(rCode, ...)                                                 \
    _checkError(rCode, __FILE__, __LINE__, ##__VA_ARGS__)

double getTime() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec / 1e6;
}

bool g_running = false;

template <class T> class GPU_Test {
  public:
    GPU_Test(int dev, bool doubles, bool tensors)
        : d_devNumber(dev), d_doubles(doubles), d_tensors(tensors) {
        checkError(cuDeviceGet(&d_dev, d_devNumber));
        checkError(cuCtxCreate(&d_ctx, 0, d_dev));

        bind();

        checkError(cublasCreate(&d_cublas), "init");

        if (d_tensors)
            checkError(cublasSetMathMode(d_cublas, CUBLAS_TENSOR_OP_MATH));

        checkError(cuMemAllocHost((void **)&d_faultyElemsHost, sizeof(int)));
        d_error = 0;

        g_running = true;

        struct sigaction action;
        memset(&action, 0, sizeof(struct sigaction));
        action.sa_handler = termHandler;
        sigaction(SIGTERM, &action, NULL);
    }

    static void termHandler(int signum) { g_running = false; }

    unsigned long long int getErrors() {
        if (*d_faultyElemsHost) {
            d_error += (long long int)*d_faultyElemsHost;
        }
        unsigned long long int tempErrs = d_error;
        d_error = 0;
        return tempErrs;
    }

    size_t getIters() { return d_iters; }

    void bind() { checkError(cuCtxSetCurrent(d_ctx), "Bind CTX"); }

    size_t totalMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(cuMemGetInfo(&freeMem, &totalMem));
        return totalMem;
    }

    size_t availMemory() {
        bind();
        size_t freeMem, totalMem;
        checkError(cuMemGetInfo(&freeMem, &totalMem));
        return freeMem;
    }

    void initBuffers(T *A, T *B, ssize_t useBytes = 0) {
        bind();

        if (useBytes == 0)
            useBytes = (ssize_t)((double)availMemory() * USEMEM);
        if (useBytes < 0)
            useBytes = (ssize_t)((double)availMemory() * (-useBytes / 100.0));

        size_t d_resultSize = sizeof(T) * SIZE * SIZE;
        d_iters = (useBytes - 2 * d_resultSize) /
                  d_resultSize; // We remove A and B sizes
        if ((size_t)useBytes < 3 * d_resultSize)
            throw std::string("Low mem for result. aborting.\n");
        checkError(cuMemAlloc(&d_Cdata, d_iters * d_resultSize), "C alloc");
        checkError(cuMemAlloc(&d_Adata, d_resultSize), "A alloc");
        checkError(cuMemAlloc(&d_Bdata, d_resultSize), "B alloc");

        checkError(cuMemAlloc(&d_faultyElemData, sizeof(int)), "faulty data");

        // Populating matrices A and B
        checkError(cuMemcpyHtoD(d_Adata, A, d_resultSize), "A -> device");
        checkError(cuMemcpyHtoD(d_Bdata, B, d_resultSize), "B -> device");
    }

    void compute() {
        bind();
        static const float alpha = 1.0f;
        static const float beta = 0.0f;
        static const double alphaD = 1.0;
        static const double betaD = 0.0;

        for (size_t i = 0; i < d_iters; ++i) {
            if (d_doubles)
                checkError(
                    cublasDgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N, SIZE, SIZE,
                                SIZE, &alphaD, (const double *)d_Adata, SIZE,
                                (const double *)d_Bdata, SIZE, &betaD,
                                (double *)d_Cdata + i * SIZE * SIZE, SIZE),
                    "DGEMM");
            else
                checkError(
                    cublasSgemm(d_cublas, CUBLAS_OP_N, CUBLAS_OP_N, SIZE, SIZE,
                                SIZE, &alpha, (const float *)d_Adata, SIZE,
                                (const float *)d_Bdata, SIZE, &beta,
                                (float *)d_Cdata + i * SIZE * SIZE, SIZE),
                    "SGEMM");
        }
    }

    bool shouldRun() { return g_running; }

  private:
    bool d_doubles;
    bool d_tensors;
    int d_devNumber;
    const char *d_kernelFile;
    size_t d_iters;
    size_t d_resultSize;

    long long int d_error;

    static const int g_blockSize = 16;

    CUdevice d_dev;
    CUcontext d_ctx;
    CUmodule d_module;
    CUfunction d_function;

    CUdeviceptr d_Cdata;
    CUdeviceptr d_Adata;
    CUdeviceptr d_Bdata;
    CUdeviceptr d_faultyElemData;
    int *d_faultyElemsHost;

    cublasHandle_t d_cublas;
};

// Returns the number of devices
int initCuda() {
    try {
        checkError(cuInit(0));
        log("init cuda");
    } catch (std::runtime_error e) {
        fprintf(stderr, "Couldn't init CUDA: %s\n", e.what());
        return 0;
    }
    int deviceCount = 0;
    checkError(cuDeviceGetCount(&deviceCount));

    if (!deviceCount)
        throw std::string("No CUDA devices");

#ifdef USEDEV
    if (USEDEV >= deviceCount)
        throw std::string("Not enough devices for USEDEV");
#endif

    return deviceCount;
}

template <class T>
void startBurn(int index, int writeFd, T *A, T *B, bool doubles, bool tensors,
               ssize_t useBytes) {
    GPU_Test<T> *our;
    try {
        our = new GPU_Test<T>(index, doubles, tensors);
        log("init test");
        our->initBuffers(A, B, useBytes);
        log("init buffers");
    } catch (const std::exception &e) {
        fprintf(stderr, "Couldn't init a GPU test: %s\n", e.what());
        exit(EMEDIUMTYPE);
    }

    // The actual work
    try {
        int eventIndex = 0;
        const int maxEvents = 2;
        CUevent events[maxEvents];
        for (int i = 0; i < maxEvents; ++i)
            cuEventCreate(events + i, 0);

        int nonWorkIters = maxEvents;

        log("start burn loop");
        while (our->shouldRun()) {
            our->compute();
            checkError(cuEventRecord(events[eventIndex], 0), "Record event");

            eventIndex = ++eventIndex % maxEvents;

            while (cuEventQuery(events[eventIndex]) != CUDA_SUCCESS)
                usleep(1000);

            if (--nonWorkIters > 0)
                continue;

            int ops = our->getIters();
            write(writeFd, &ops, sizeof(int));
            ops = our->getErrors();
            write(writeFd, &ops, sizeof(int));
        }
        log("end burn loop");

        for (int i = 0; i < maxEvents; ++i)
            // cuEventSynchronize(events[i]);
            cuEventDestroy(events[i]);
        log("start test cleanup");
        delete our;
        log("test cleanup complete");
    } catch (const std::exception &e) {
        fprintf(stderr, "Failure during compute: %s\n", e.what());
        int ops = -1;
        // Signalling that we failed
        write(writeFd, &ops, sizeof(int));
        write(writeFd, &ops, sizeof(int));
        exit(ECONNREFUSED);
    }
}

template <class T>
void launch(int runLength, bool useDoubles, bool useTensorCores,
            ssize_t useBytes, int device_id) {
    struct sigaction action;
    memset(&action, 0, sizeof(struct sigaction));
    action.sa_handler = terminateSelf;
    sigaction(SIGTERM, &action, NULL);

    time_t startTime = time(0);

    log("launch");
    // Initting A and B with random data
    T *A = (T *)malloc(sizeof(T) * SIZE * SIZE);
    T *B = (T *)malloc(sizeof(T) * SIZE * SIZE);
    srand(10);
    for (size_t i = 0; i < SIZE * SIZE; ++i) {
        A[i] = (T)((double)(rand() % 1000000) / 100000.0);
        B[i] = (T)((double)(rand() % 1000000) / 100000.0);
    }

    log("init matrices");

    // Forking a process..  This one checks the number of devices to use,
    // returns the value, and continues to use the first one.
    int mainPipe[2];
    pipe(mainPipe);
    int readMain = mainPipe[0];
    std::vector<int> clientPipes;
    std::vector<pid_t> clientPids;
    clientPipes.push_back(readMain);

    if (device_id > -1) {
        pid_t myPid = fork();
        if (!myPid) {
            // Child
            close(mainPipe[0]);
            int writeFd = mainPipe[1];
            initCuda();
            int devCount = 1;
            write(writeFd, &devCount, sizeof(int));
            startBurn<T>(device_id, writeFd, A, B, useDoubles, useTensorCores,
                         useBytes);
            close(writeFd);
            return;
        } else {
            clientPids.push_back(myPid);
            close(mainPipe[1]);
            int devCount;
            read(readMain, &devCount, sizeof(int));
            log("sleeping1");
            while ((runLength == 0 && !terminateFlag) ||
                   time(0) - startTime < runLength) {
                sleep(1);
            }
        }
        for (size_t i = 0; i < clientPipes.size(); ++i)
            close(clientPipes.at(i));
    } else {
        pid_t myPid = fork();
        if (!myPid) {
            // Child
            close(mainPipe[0]);
            int writeFd = mainPipe[1];
            int devCount = initCuda();
            write(writeFd, &devCount, sizeof(int));

            startBurn<T>(0, writeFd, A, B, useDoubles, useTensorCores,
                         useBytes);

            close(writeFd);
            return;
        } else {
            clientPids.push_back(myPid);

            close(mainPipe[1]);
            int devCount;
            read(readMain, &devCount, sizeof(int));

            if (!devCount) {
                fprintf(stderr, "No CUDA devices\n");
                exit(ENODEV);
            } else {
                for (int i = 1; i < devCount; ++i) {
                    int slavePipe[2];
                    pipe(slavePipe);
                    clientPipes.push_back(slavePipe[0]);

                    pid_t slavePid = fork();

                    if (!slavePid) {
                        // Child
                        close(slavePipe[0]);
                        initCuda();
                        startBurn<T>(i, slavePipe[1], A, B, useDoubles,
                                     useTensorCores, useBytes);

                        close(slavePipe[1]);
                        return;
                    } else {
                        clientPids.push_back(slavePid);
                        close(slavePipe[1]);
                    }
                }

                log("sleeping2");
                while ((runLength == 0 && !terminateFlag) ||
                       time(0) - startTime < runLength) {
                    sleep(1);
                }
            }
        }
        for (size_t i = 0; i < clientPipes.size(); ++i)
            close(clientPipes.at(i));
    }
    log("killing processes");
    for (size_t i = 0; i < clientPids.size(); ++i) {
        kill(clientPids.at(i), SIGKILL);
    }
}

void showHelp() {
    printf("GPU Burn\n");
    printf("Usage: gpu-burn [OPTIONS] [TIME]\n\n");
    printf("-m X\tUse X MB of memory.\n");
    printf("-m N%%\tUse N%% of the available GPU memory.  Default is %d%%\n",
           (int)(USEMEM * 100));
    printf("-d\tUse doubles\n");
    printf("-tc\tTry to use Tensor cores\n");
    printf("-l\tLists all GPUs in the system\n");
    printf("-i N\tExecute only on GPU N\n");
    printf("-h\tShow this help message\n\n");
    printf("Examples:\n");
    printf("  gpu-burn -d 3600 # burns all GPUs with doubles for an hour\n");
    printf(
        "  gpu-burn -m 50%% # burns using 50%% of the available GPU memory\n");
    printf("  gpu-burn -l # list GPUs\n");
    printf("  gpu-burn -i 2 # burns only GPU of index 2\n");
}

// NNN MB
// NN% <0
// 0 --- error
ssize_t decodeUSEMEM(const char *s) {
    char *s2;
    int64_t r = strtoll(s, &s2, 10);
    if (s == s2)
        return 0;
    if (*s2 == '%')
        return (s2[1] == 0) ? -r : 0;
    return (*s2 == 0) ? r * 1024 * 1024 : 0;
}

int main(int argc, char **argv) {
    int runLength = 0;
    bool useDoubles = false;
    bool useTensorCores = false;
    int thisParam = 0;
    ssize_t useBytes = 0; // 0 == use USEMEM% of free mem
    int device_id = -1;

    std::vector<std::string> args(argv, argv + argc);
    for (size_t i = 1; i < args.size(); ++i) {
        if (argc >= 2 && std::string(argv[i]).find("-h") != std::string::npos) {
            showHelp();
            return 0;
        }
        if (argc >= 2 && std::string(argv[i]).find("-l") != std::string::npos) {
            int count = initCuda();
            if (count == 0) {
                throw std::runtime_error("No CUDA capable GPUs found.\n");
            }
            for (int i_dev = 0; i_dev < count; i_dev++) {
                CUdevice device_l;
                char device_name[255];
                checkError(cuDeviceGet(&device_l, i_dev));
                checkError(cuDeviceGetName(device_name, 255, device_l));
                size_t device_mem_l;
                checkError(cuDeviceTotalMem(&device_mem_l, device_l));
            }
            thisParam++;
            return 0;
        }
        if (argc >= 2 && std::string(argv[i]).find("-d") != std::string::npos) {
            useDoubles = true;
            thisParam++;
        }
        if (argc >= 2 &&
            std::string(argv[i]).find("-tc") != std::string::npos) {
            useTensorCores = true;
            thisParam++;
        }
        if (argc >= 2 && strncmp(argv[i], "-m", 2) == 0) {
            thisParam++;

            // -mNNN[%]
            // -m NNN[%]
            if (argv[i][2]) {
                useBytes = decodeUSEMEM(argv[i] + 2);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                useBytes = decodeUSEMEM(argv[i]);
            } else {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
            if (useBytes == 0) {
                fprintf(stderr, "Syntax error near -m\n");
                exit(EINVAL);
            }
        }
        if (argc >= 2 && strncmp(argv[i], "-i", 2) == 0) {
            thisParam++;

            if (argv[i][2]) {
                device_id = strtol(argv[i] + 2, NULL, 0);
            } else if (i + 1 < args.size()) {
                i++;
                thisParam++;
                device_id = strtol(argv[i], NULL, 0);
            } else {
                fprintf(stderr, "Syntax error near -i\n");
                exit(EINVAL);
            }
        }
    }

    if (argc - thisParam < 2) {
        log("Run length not specified in the command line. \nBurning "
            "Indefinitely");
    } else {
        runLength = atoi(argv[1 + thisParam]);
        log("Burning for " + std::to_string(runLength) + " seconds.");
    }

    if (useDoubles)
        launch<double>(runLength, useDoubles, useTensorCores, useBytes,
                       device_id);
    else
        launch<float>(runLength, useDoubles, useTensorCores, useBytes,
                      device_id);

    return 0;
}
