// SPDX-License-Identifier: Apache-2.0
#include <stdarg.h>
#include <stdio.h>
#include "trace.h"
char dc_log[16384];
volatile unsigned dc_log_position;
int trace_printf(const char *format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    int n = vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    for (int i = 0; buffer[i]; ++i) {
        dc_log[dc_log_position++ % sizeof(dc_log)] = buffer[i];
    }
    printf("%s", buffer);
    return n;
}
