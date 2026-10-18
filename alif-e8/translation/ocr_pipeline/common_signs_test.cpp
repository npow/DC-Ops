#include "common_signs.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

void match(const char* input, const char* expected) {
    const char* actual = dcops::lookup_common_sign(input);
    assert(actual && std::strcmp(actual, expected) == 0);
}
int main() {
    assert(dcops::common_sign_count() >= 40 && dcops::common_sign_count() <= 60);
    match("STOP", "ALTO"); match("Stop", "ALTO"); match("stop!!!", "ALTO");
    match("  emergency\t EXIT\n", "SALIDA DE EMERGENCIA");
    match("no-entry", "PROHIBIDO EL PASO"); match("(No smoking.)", "PROHIBIDO FUMAR");
    match("Please   close the door", "POR FAVOR CIERRE LA PUERTA");
    match("HELP", "AYUDA"); match("EXIT", "SALIDA"); match("HELLO", "HOLA");
    for (const char* input : {"", "   !!!", "STOPWATCH", "ST0P", "S T O P", "S.T.O.P", "EX1T",
                             "STOP NOW", "NO/ENTRY", "hello world", "caf\xc3\xa9", "STOP\x01"})
        assert(dcops::lookup_common_sign(input) == nullptr);
    assert(dcops::lookup_common_sign(nullptr) == nullptr);
    std::string oversized(300, 'A'); assert(!dcops::lookup_common_sign(oversized.c_str()));
    std::cout << "{\"sign_count\":" << dcops::common_sign_count()
              << ",\"normalization_cases\":10,\"unsupported_cases\":14}\n";
}
