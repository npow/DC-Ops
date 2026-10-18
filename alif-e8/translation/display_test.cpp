// SPDX-License-Identifier: Apache-2.0
#include "translation_display.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>
int main() {
    constexpr int sw=320,sh=240,w=480,h=800,pad=32;
    std::vector<uint16_t> camera(sw*sh), output(w*h+pad*2,0xbeef);
    for (int y=0;y<sh;++y) for (int x=0;x<sw;++x) camera[y*sw+x]=(x<<8)^y;
    dcops::Image input{camera.data(),sw,sh,sw};
    dcops::Surface surface{output.data()+pad,w,h,w};
    auto crop=dcops::camera_viewport(sw,sh,w,h);
    assert(crop.x==88 && crop.y==0 && crop.width==144 && crop.height==240);
    assert(dcops::render_translation(surface,input,nullptr,0));
    assert(surface.pixels[0]==camera[88]);
    assert(surface.pixels[(h-1)*w+w-1]==camera[239*sw+231]);
    const auto video=output;
    dcops::TranslationBox invisible{-100,-100,20,20,"invisible"};
    assert(dcops::render_translation(surface,input,&invisible,1));
    assert(video==output);
    dcops::TranslationBox transparent{91,90,228,150,"ALTO"};
    assert(dcops::render_translation(surface,input,&transparent,1));
    assert(surface.pixels[480*w+240]==camera[144*sw+160]);
    assert(surface.pixels[480*w+400]==camera[144*sw+208]);
    std::fill(camera.begin(),camera.end(),0xffff);
    dcops::TranslationBox scan{91,90,228,150,nullptr,0xffe0};
    assert(dcops::render_translation(surface,input,&scan,1));
    assert(surface.pixels[300*w+10]==0xffe0);
    assert(surface.pixels[400*w+240]==0xffff);
    assert(surface.pixels[499*w+240]==0xffe0);
    dcops::TranslationBox accents{105,50,210,75,u8"¡Atención! ¿Dónde está el niño? Sí, pingüino."};
    assert(dcops::render_translation(surface,input,&accents,1));
    assert(std::count(surface.pixels,surface.pixels+w*h,uint16_t(0))>100);
    auto accented=output;
    accents.translated_utf8="!Atencion! ?Donde esta el nino? Si, pinguino.";
    assert(dcops::render_translation(surface,input,&accents,1));
    assert(accented!=output);
    dcops::TranslationBox malformed{-1e30f,-1e30f,1e30f,1e30f,"\xc0\xff\xe2"};
    dcops::DisplayStats stats{123,2,4,90,300,100,-1};
    assert(dcops::render_translation(surface,input,&malformed,1,&stats));
    malformed.x0=NAN;
    assert(dcops::render_translation(surface,input,&malformed,1));
    for (int i=0;i<pad;++i) assert(output[i]==0xbeef && output[pad+w*h+i]==0xbeef);
    assert(!dcops::render_translation(surface,input,nullptr,1));
    assert(!dcops::render_translation({nullptr,w,h,w},input,nullptr,0));
}
