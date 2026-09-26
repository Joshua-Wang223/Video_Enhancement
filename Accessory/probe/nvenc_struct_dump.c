/**
 * NVENC SDK 结构体 & 函数表 dump — C 程序
 *
 * 目的: 用真实 NVENC SDK header 获取准确的 sizeof/offset/函数索引，
 *       消除 ctypes 猜测。
 *
 * 编译 (需要 nvEncodeAPI.h):
 *   gcc -o test_nvenc_struct_dump test_nvenc_struct_dump.c \
 *       -I<path-to-nvEncodeAPI.h>
 *
 * 若系统未安装 header，先下载 FFmpeg nv-codec-headers:
 *   git clone --depth 1 --branch n13.0.19.0 \
 *       https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers
 *   gcc -o test_nvenc_struct_dump test_nvenc_struct_dump.c \
 *       -I/tmp/nv-codec-headers/include
 *
 * 运行:
 *   ./test_nvenc_struct_dump
 */

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

/* 尝试多个可能的 header 路径 */
#if __has_include("nvEncodeAPI.h")
  #include "nvEncodeAPI.h"
  #define HEADER_FOUND 1
#elif __has_include("ffnvcodec/nvEncodeAPI.h")
  #include "ffnvcodec/nvEncodeAPI.h"
  #define HEADER_FOUND 1
#else
  #define HEADER_FOUND 0
  /* 手写关键结构体以便至少能编译并提示用户 */
  #warning "nvEncodeAPI.h not found — will only print manual struct info"
#endif

int main() {
    printf("=== NVENC SDK Struct Dump ===\n");
    printf("Compiler: " __VERSION__ "\n");
    printf("sizeof(void*): %zu\n\n", sizeof(void*));

#if HEADER_FOUND
    /* ================================================================
     * NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS
     * ================================================================ */
    {
        NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS p;
        printf("=== NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS ===\n");
        printf("sizeof:                  %zu (0x%zx)\n", sizeof(p), sizeof(p));
        printf("offset.version:          %zu\n", offsetof(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS, version));
        printf("offset.deviceType:       %zu\n", offsetof(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS, deviceType));
        printf("offset.device:           %zu\n", offsetof(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS, device));
        printf("offset.reserved:         %zu\n", offsetof(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS, reserved));
        printf("offset.apiVersion:       %zu\n", offsetof(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS, apiVersion));

        /* 探测 reserved1 偏移 (不同 SDK 版本字段名可能不同) */
#ifdef NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER
        printf("\nNV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER: 0x%08x\n",
               NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER);
#endif
    }

    /* ================================================================
     * NV_ENCODE_API_FUNCTION_LIST
     * ================================================================ */
    {
        NV_ENCODE_API_FUNCTION_LIST flist;
        printf("\n=== NV_ENCODE_API_FUNCTION_LIST ===\n");
        printf("sizeof: %zu (0x%zx)\n", sizeof(flist), sizeof(flist));
        printf("offset.version: %zu\n", offsetof(NV_ENCODE_API_FUNCTION_LIST, version));

#ifdef NV_ENCODE_API_FUNCTION_LIST_VER
        printf("NV_ENCODE_API_FUNCTION_LIST_VER: 0x%08x\n",
               NV_ENCODE_API_FUNCTION_LIST_VER);
#endif
    }

    /* ================================================================
     * NV_ENC_INITIALIZE_PARAMS
     * ================================================================ */
    {
        NV_ENC_INITIALIZE_PARAMS p;
        printf("\n=== NV_ENC_INITIALIZE_PARAMS ===\n");
        printf("sizeof: %zu (0x%zx)\n", sizeof(p), sizeof(p));
        printf("offset.version:          %zu\n", offsetof(NV_ENC_INITIALIZE_PARAMS, version));
        printf("offset.encodeGUID:       %zu\n", offsetof(NV_ENC_INITIALIZE_PARAMS, encodeGUID));
        printf("offset.presetGUID:       %zu\n", offsetof(NV_ENC_INITIALIZE_PARAMS, presetGUID));
        printf("offset.encodeWidth:      %zu\n", offsetof(NV_ENC_INITIALIZE_PARAMS, encodeWidth));
        printf("offset.encodeHeight:     %zu\n", offsetof(NV_ENC_INITIALIZE_PARAMS, encodeHeight));
        printf("offset.darWidth:         %zu\n", offsetof(NV_ENC_INITIALIZE_PARAMS, darWidth));
        printf("offset.darHeight:        %zu\n", offsetof(NV_ENC_INITIALIZE_PARAMS, darHeight));
        printf("offset.frameRateNum:     %zu\n", offsetof(NV_ENC_INITIALIZE_PARAMS, frameRateNum));
        printf("offset.frameRateDen:     %zu\n", offsetof(NV_ENC_INITIALIZE_PARAMS, frameRateDen));
        printf("offset.enableEncodeAsync:%zu\n", offsetof(NV_ENC_INITIALIZE_PARAMS, enableEncodeAsync));
        printf("offset.enablePTD:        %zu\n", offsetof(NV_ENC_INITIALIZE_PARAMS, enablePTD));

#ifdef NV_ENC_INITIALIZE_PARAMS_VER
        printf("\nNV_ENC_INITIALIZE_PARAMS_VER: 0x%08x\n",
               NV_ENC_INITIALIZE_PARAMS_VER);
#endif
    }

    /* ================================================================
     * 其他关键结构体 sizeof
     * ================================================================ */
    {
        printf("\n=== 其他结构体 sizeof ===\n");
#define PRINT_SIZEOF(T) printf("sizeof(" #T "): %zu (0x%zx)\n", sizeof(T), sizeof(T))

        PRINT_SIZEOF(NV_ENC_PRESET_CONFIG);
        PRINT_SIZEOF(NV_ENC_CREATE_INPUT_BUFFER);
        PRINT_SIZEOF(NV_ENC_CREATE_BITSTREAM_BUFFER);
        PRINT_SIZEOF(NV_ENC_CREATE_MV_BUFFER);
        PRINT_SIZEOF(NV_ENC_LOCK_BITSTREAM);
        PRINT_SIZEOF(NV_ENC_LOCK_INPUT_BUFFER);
        PRINT_SIZEOF(NV_ENC_MAP_INPUT_RESOURCE);
        PRINT_SIZEOF(NV_ENC_PIC_PARAMS);
        PRINT_SIZEOF(NV_ENC_REGISTER_RESOURCE);
        PRINT_SIZEOF(NV_ENC_CONFIG);

#undef PRINT_SIZEOF
    }

    /* ================================================================
     * Phase 3 关键结构体字段偏移
     * ================================================================ */
    {
        printf("\n=== NV_ENC_CREATE_INPUT_BUFFER 字段偏移 ===\n");
#define PRINT_OFFSET(T, field) printf("  %-40s offset=%3zu\n", #field, offsetof(T, field))
        PRINT_OFFSET(NV_ENC_CREATE_INPUT_BUFFER, version);
        PRINT_OFFSET(NV_ENC_CREATE_INPUT_BUFFER, width);
        PRINT_OFFSET(NV_ENC_CREATE_INPUT_BUFFER, height);
        PRINT_OFFSET(NV_ENC_CREATE_INPUT_BUFFER, memoryHeap);
        PRINT_OFFSET(NV_ENC_CREATE_INPUT_BUFFER, bufferFmt);
        PRINT_OFFSET(NV_ENC_CREATE_INPUT_BUFFER, reserved);
        PRINT_OFFSET(NV_ENC_CREATE_INPUT_BUFFER, inputBuffer);
        PRINT_OFFSET(NV_ENC_CREATE_INPUT_BUFFER, pSysMemBuffer);
#ifdef NV_ENC_CREATE_INPUT_BUFFER_VER
        printf("\nNV_ENC_CREATE_INPUT_BUFFER_VER: 0x%08x\n",
               NV_ENC_CREATE_INPUT_BUFFER_VER);
#endif

        printf("\n=== NV_ENC_LOCK_INPUT_BUFFER 字段偏移 ===\n");
        PRINT_OFFSET(NV_ENC_LOCK_INPUT_BUFFER, version);
        PRINT_OFFSET(NV_ENC_LOCK_INPUT_BUFFER, inputBuffer);
        PRINT_OFFSET(NV_ENC_LOCK_INPUT_BUFFER, bufferDataPtr);
        PRINT_OFFSET(NV_ENC_LOCK_INPUT_BUFFER, pitch);
#ifdef NV_ENC_LOCK_INPUT_BUFFER_VER
        printf("\nNV_ENC_LOCK_INPUT_BUFFER_VER: 0x%08x\n",
               NV_ENC_LOCK_INPUT_BUFFER_VER);
#endif

        printf("\n=== NV_ENC_CREATE_BITSTREAM_BUFFER 字段偏移 ===\n");
        PRINT_OFFSET(NV_ENC_CREATE_BITSTREAM_BUFFER, version);
        PRINT_OFFSET(NV_ENC_CREATE_BITSTREAM_BUFFER, bitstreamBuffer);
#ifdef NV_ENC_CREATE_BITSTREAM_BUFFER_VER
        printf("\nNV_ENC_CREATE_BITSTREAM_BUFFER_VER: 0x%08x\n",
               NV_ENC_CREATE_BITSTREAM_BUFFER_VER);
#endif

        printf("\n=== NV_ENC_LOCK_BITSTREAM 字段偏移 ===\n");
        PRINT_OFFSET(NV_ENC_LOCK_BITSTREAM, version);
        PRINT_OFFSET(NV_ENC_LOCK_BITSTREAM, outputBitstream);
        PRINT_OFFSET(NV_ENC_LOCK_BITSTREAM, bitstreamSizeInBytes);
        PRINT_OFFSET(NV_ENC_LOCK_BITSTREAM, bitstreamBufferPtr);
#ifdef NV_ENC_LOCK_BITSTREAM_VER
        printf("\nNV_ENC_LOCK_BITSTREAM_VER: 0x%08x\n",
               NV_ENC_LOCK_BITSTREAM_VER);
#endif
#undef PRINT_OFFSET
    }

    /* ================================================================
     * 宏值
     * ================================================================ */
    {
        printf("\n=== 关键宏 ===\n");
#ifdef NVENCAPI_VERSION
        printf("NVENCAPI_VERSION: 0x%x (%d.%d)\n",
               NVENCAPI_VERSION, NVENCAPI_VERSION >> 4, NVENCAPI_VERSION & 0xF);
#endif

#define PRINT_VER(M) printf("%-50s 0x%08x\n", #M, M)

        /* 尝试打印所有已知的 VER 宏 */
#ifdef NVENCAPI_VERSION
        PRINT_VER(NVENCAPI_VERSION);
#endif
#ifdef NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER
        PRINT_VER(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER);
#endif
#ifdef NV_ENCODE_API_FUNCTION_LIST_VER
        PRINT_VER(NV_ENCODE_API_FUNCTION_LIST_VER);
#endif
#ifdef NV_ENC_INITIALIZE_PARAMS_VER
        PRINT_VER(NV_ENC_INITIALIZE_PARAMS_VER);
#endif
#ifdef NV_ENC_PRESET_CONFIG_VER
        PRINT_VER(NV_ENC_PRESET_CONFIG_VER);
#endif
#ifdef NV_ENC_CONFIG_VER
        PRINT_VER(NV_ENC_CONFIG_VER);
#endif
#ifdef NV_ENC_PIC_PARAMS_VER
        PRINT_VER(NV_ENC_PIC_PARAMS_VER);
#endif
#ifdef NV_ENC_CREATE_INPUT_BUFFER_VER
        PRINT_VER(NV_ENC_CREATE_INPUT_BUFFER_VER);
#endif
#ifdef NV_ENC_CREATE_BITSTREAM_BUFFER_VER
        PRINT_VER(NV_ENC_CREATE_BITSTREAM_BUFFER_VER);
#endif
#ifdef NV_ENC_LOCK_BITSTREAM_VER
        PRINT_VER(NV_ENC_LOCK_BITSTREAM_VER);
#endif
#ifdef NV_ENC_LOCK_INPUT_BUFFER_VER
        PRINT_VER(NV_ENC_LOCK_INPUT_BUFFER_VER);
#endif
#ifdef NV_ENC_MAP_INPUT_RESOURCE_VER
        PRINT_VER(NV_ENC_MAP_INPUT_RESOURCE_VER);
#endif
#ifdef NV_ENC_REGISTER_RESOURCE_VER
        PRINT_VER(NV_ENC_REGISTER_RESOURCE_VER);
#endif
#ifdef NV_ENC_CREATE_MV_BUFFER_VER
        PRINT_VER(NV_ENC_CREATE_MV_BUFFER_VER);
#endif

#undef PRINT_VER
    }

    /* ================================================================
     * 验证 version 计算公式
     * ================================================================ */
    {
        printf("\n=== Version 公式验证 ===\n");
        printf("假设公式: sizeof | (api_ver << 16) | (0x7 << 28)\n\n");

        uint32_t api_ver = (13 << 4) | 0;  /* 13.0 */

        {
            size_t sz = sizeof(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS);
            uint32_t computed = (uint32_t)sz | (api_ver << 16) | (0x7U << 28);
            printf("OpenEncodeSessionExParams:\n");
            printf("  sizeof=%zu (0x%zx)\n", sz, sz);
            printf("  computed_ver = 0x%08x\n", computed);
#ifdef NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER
            printf("  actual_ver   = 0x%08x\n", NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER);
            printf("  MATCH: %s\n",
                   computed == NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER ? "YES" : "NO");
#endif
        }

        {
            size_t sz = sizeof(NV_ENCODE_API_FUNCTION_LIST);
            uint32_t computed = (uint32_t)sz | (api_ver << 16) | (0x7U << 28);
            printf("\nNV_ENCODE_API_FUNCTION_LIST:\n");
            printf("  sizeof=%zu (0x%zx)\n", sz, sz);
            printf("  computed_ver = 0x%08x\n", computed);
#ifdef NV_ENCODE_API_FUNCTION_LIST_VER
            printf("  actual_ver   = 0x%08x\n", NV_ENCODE_API_FUNCTION_LIST_VER);
            printf("  MATCH: %s\n",
                   computed == NV_ENCODE_API_FUNCTION_LIST_VER ? "YES" : "NO");
#endif
        }
    }

    /* ================================================================
     * 函数索引 (NVENC API Function Table)
     *
     * nv-codec-headers 用 struct 具名字段 (不用 enum),
     * 所以无法直接用 enum 名打印索引。
     * 改为打印 struct 字段偏移，再由偏移计算索引:
     *   index = (offset - 8) / sizeof(void*)
     * ================================================================ */
    {
        printf("\n=== NVENC API Function Table Indices ===\n");
        printf("nv-codec-headers: 索引 = (字段偏移 - 8) / sizeof(void*)\n");
        printf("sizeof(void*) = %zu\n\n", sizeof(void*));

        NV_ENCODE_API_FUNCTION_LIST fl;
        size_t ptr_base = offsetof(NV_ENCODE_API_FUNCTION_LIST, nvEncOpenEncodeSession);

#define PRINT_FUNC_OFFSET(name) \
    do { \
        size_t _off = offsetof(NV_ENCODE_API_FUNCTION_LIST, name); \
        size_t _idx = (_off - ptr_base) / sizeof(void*); \
        printf("  [%3zu] %-42s offset=%3zu\n", _idx, #name, _off); \
    } while(0)

        PRINT_FUNC_OFFSET(nvEncOpenEncodeSession);
        PRINT_FUNC_OFFSET(nvEncGetEncodeGUIDCount);
        PRINT_FUNC_OFFSET(nvEncGetEncodeProfileGUIDCount);
        PRINT_FUNC_OFFSET(nvEncGetEncodeProfileGUIDs);
        PRINT_FUNC_OFFSET(nvEncGetEncodeGUIDs);
        PRINT_FUNC_OFFSET(nvEncGetInputFormatCount);
        PRINT_FUNC_OFFSET(nvEncGetInputFormats);
        PRINT_FUNC_OFFSET(nvEncGetEncodeCaps);
        PRINT_FUNC_OFFSET(nvEncGetEncodePresetCount);
        PRINT_FUNC_OFFSET(nvEncGetEncodePresetGUIDs);
        PRINT_FUNC_OFFSET(nvEncGetEncodePresetConfig);
        PRINT_FUNC_OFFSET(nvEncInitializeEncoder);
        PRINT_FUNC_OFFSET(nvEncCreateInputBuffer);
        PRINT_FUNC_OFFSET(nvEncDestroyInputBuffer);
        PRINT_FUNC_OFFSET(nvEncCreateBitstreamBuffer);
        PRINT_FUNC_OFFSET(nvEncDestroyBitstreamBuffer);
        PRINT_FUNC_OFFSET(nvEncEncodePicture);
        PRINT_FUNC_OFFSET(nvEncLockBitstream);
        PRINT_FUNC_OFFSET(nvEncUnlockBitstream);
        PRINT_FUNC_OFFSET(nvEncLockInputBuffer);
        PRINT_FUNC_OFFSET(nvEncUnlockInputBuffer);
        PRINT_FUNC_OFFSET(nvEncGetEncodeStats);
        PRINT_FUNC_OFFSET(nvEncGetSequenceParams);
        PRINT_FUNC_OFFSET(nvEncRegisterAsyncEvent);
        PRINT_FUNC_OFFSET(nvEncUnregisterAsyncEvent);
        PRINT_FUNC_OFFSET(nvEncMapInputResource);
        PRINT_FUNC_OFFSET(nvEncUnmapInputResource);
        PRINT_FUNC_OFFSET(nvEncDestroyEncoder);
        PRINT_FUNC_OFFSET(nvEncInvalidateRefFrames);
        PRINT_FUNC_OFFSET(nvEncOpenEncodeSessionEx);
        PRINT_FUNC_OFFSET(nvEncRegisterResource);
        PRINT_FUNC_OFFSET(nvEncUnregisterResource);
        PRINT_FUNC_OFFSET(nvEncReconfigureEncoder);
        PRINT_FUNC_OFFSET(nvEncCreateMVBuffer);
        PRINT_FUNC_OFFSET(nvEncDestroyMVBuffer);
        PRINT_FUNC_OFFSET(nvEncRunMotionEstimationOnly);
        PRINT_FUNC_OFFSET(nvEncGetLastErrorString);
        PRINT_FUNC_OFFSET(nvEncSetIOCudaStreams);
        PRINT_FUNC_OFFSET(nvEncGetEncodePresetConfigEx);
        PRINT_FUNC_OFFSET(nvEncGetSequenceParamEx);
        PRINT_FUNC_OFFSET(nvEncRestoreEncoderState);
        PRINT_FUNC_OFFSET(nvEncLookaheadPicture);

#undef PRINT_FUNC_OFFSET

        printf("\n  — reserved1 at offset %zu (index %zu) —\n",
               offsetof(NV_ENCODE_API_FUNCTION_LIST, reserved1),
               (offsetof(NV_ENCODE_API_FUNCTION_LIST, reserved1) - ptr_base) / sizeof(void*));
        printf("  — reserved2[275] starts at offset %zu (indices %zu-%zu) —\n",
               offsetof(NV_ENCODE_API_FUNCTION_LIST, reserved2),
               (offsetof(NV_ENCODE_API_FUNCTION_LIST, reserved2) - ptr_base) / sizeof(void*),
               (sizeof(fl) - ptr_base) / sizeof(void*) - 1);
    }

    /* ================================================================
     * 运行时验证: 尝试用正确索引调用 NvEncOpenEncodeSessionEx
     * ================================================================ */
#ifdef NvEncOpenEncodeSessionEx
    {
        printf("\n=== 运行时: 用 C SDK 调用 NvEncOpenEncodeSessionEx ===\n");
        printf("(需要链接 -lnvidia-encode)\n");
        printf("正确索引: %d\n", NvEncOpenEncodeSessionEx);
        printf("struct_ver: 0x%08x\n", NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER);
    }
#endif

#else  /* !HEADER_FOUND */
    printf("nvEncodeAPI.h 未找到。\n\n");
    printf("请先安装 NVENC SDK header:\n");
    printf("  git clone --depth 1 --branch n13.0.19.0 \\\n");
    printf("      https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers\n");
    printf("  gcc -o test_nvenc_struct_dump test_nvenc_struct_dump.c \\\n");
    printf("      -I/tmp/nv-codec-headers/include\n");
#endif

    return 0;
}
