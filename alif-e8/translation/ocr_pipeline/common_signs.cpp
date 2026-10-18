#include "common_signs.h"
#include <cstring>

namespace dcops {
namespace {
struct Sign { const char* english; const char* spanish; };
constexpr Sign signs[] = {
    {"STOP", "ALTO"}, {"EXIT", "SALIDA"}, {"ENTRANCE", "ENTRADA"},
    {"ENTER", "ENTRAR"}, {"NO ENTRY", "PROHIBIDO EL PASO"},
    {"DO NOT ENTER", "PROHIBIDO EL PASO"}, {"NO EXIT", "SIN SALIDA"},
    {"EXIT ONLY", "SOLO SALIDA"}, {"EMERGENCY EXIT", "SALIDA DE EMERGENCIA"},
    {"FIRE EXIT", "SALIDA DE EMERGENCIA"}, {"PUSH", "EMPUJAR"},
    {"PULL", "TIRAR"}, {"PUSH TO OPEN", "EMPUJE PARA ABRIR"},
    {"PULL TO OPEN", "TIRE PARA ABRIR"}, {"OPEN", "ABIERTO"},
    {"CLOSED", "CERRADO"}, {"WELCOME", "BIENVENIDOS"}, {"HELLO", "HOLA"},
    {"HELP", "AYUDA"}, {"INFORMATION", "INFORMACIÓN"},
    {"DANGER", "PELIGRO"}, {"WARNING", "ADVERTENCIA"}, {"CAUTION", "PRECAUCIÓN"},
    {"HIGH VOLTAGE", "ALTA TENSIÓN"}, {"DO NOT TOUCH", "NO TOCAR"},
    {"HOT SURFACE", "SUPERFICIE CALIENTE"}, {"WET FLOOR", "SUELO MOJADO"},
    {"NO SMOKING", "PROHIBIDO FUMAR"}, {"NO PARKING", "PROHIBIDO ESTACIONAR"},
    {"PARKING", "ESTACIONAMIENTO"}, {"KEEP OUT", "PROHIBIDO EL PASO"},
    {"STAFF ONLY", "SOLO PERSONAL"}, {"AUTHORIZED PERSONNEL ONLY", "SOLO PERSONAL AUTORIZADO"},
    {"RESTROOM", "BAÑO"}, {"RESTROOMS", "BAÑOS"}, {"TOILET", "BAÑO"},
    {"TOILETS", "BAÑOS"}, {"MEN", "HOMBRES"}, {"WOMEN", "MUJERES"},
    {"FIRST AID", "PRIMEROS AUXILIOS"}, {"FIRE EXTINGUISHER", "EXTINTOR"},
    {"DRINKING WATER", "AGUA POTABLE"}, {"DO NOT DRINK", "NO BEBER"},
    {"OUT OF SERVICE", "FUERA DE SERVICIO"}, {"OUT OF ORDER", "FUERA DE SERVICIO"},
    {"PLEASE WAIT", "POR FAVOR ESPERE"}, {"WAIT HERE", "ESPERE AQUÍ"},
    {"KEEP DOOR CLOSED", "MANTENGA LA PUERTA CERRADA"},
    {"KEEP THIS DOOR CLOSED", "MANTENGA ESTA PUERTA CERRADA"},
    {"PLEASE CLOSE THE DOOR", "POR FAVOR CIERRE LA PUERTA"},
    {"WASH YOUR HANDS", "LÁVESE LAS MANOS"}, {"NO FOOD OR DRINK", "PROHIBIDO COMER Y BEBER"},
    {"NO PHOTOGRAPHY", "PROHIBIDO TOMAR FOTOS"}, {"TURN LEFT", "GIRE A LA IZQUIERDA"},
    {"TURN RIGHT", "GIRE A LA DERECHA"}, {"ELEVATOR", "ASCENSOR"},
    {"STAIRS", "ESCALERAS"}, {"RECEPTION", "RECEPCIÓN"},
    {"THANK YOU", "GRACIAS"}, {"PLEASE WAIT HERE", "POR FAVOR ESPERE AQUÍ"},
};
bool normalize(const char* text, char output[128]) {
    if (!text) return false;
    size_t used = 0;
    bool space = false;
    for (size_t i = 0; i < 256; ++i) {
        unsigned char c = text[i];
        if (!c) { output[used] = 0; return used != 0; }
        if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
        if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
            if (space && used) { if (used >= 127) return false; output[used++] = ' '; }
            if (used >= 127) return false;
            output[used++] = char(c); space = false;
        } else if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '-' ||
                   c == '.' || c == ',' || c == ':' || c == ';' || c == '!' || c == '?' ||
                   c == '(' || c == ')' || c == '"') space = true;
        else return false;
    }
    return false;
}
}
const char* lookup_common_sign(const char* english) {
    char normalized[128];
    if (!normalize(english, normalized)) return nullptr;
    for (const auto& sign : signs)
        if (std::strcmp(normalized, sign.english) == 0) return sign.spanish;
    return nullptr;
}
size_t common_sign_count() { return sizeof(signs) / sizeof(signs[0]); }
}
