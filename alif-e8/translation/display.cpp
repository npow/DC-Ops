// SPDX-License-Identifier: Apache-2.0
#include "translation_display.h"
#include "../board/DevKit-E8/font.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace dcops {
namespace {
struct Rect { int x0, y0, x1, y1; };
void fill(Surface s, Rect r, uint16_t color) {
    for (int y = std::max(0, r.y0); y < std::min(s.height, r.y1); ++y)
        for (int x = std::max(0, r.x0); x < std::min(s.width, r.x1); ++x)
            s.pixels[y*s.stride+x] = color;
}
unsigned next_character(const char *&p, const char *end) {
    if (p == end || !*p) return 0;
    const unsigned a = static_cast<unsigned char>(*p++);
    if (a < 128) return a;
    unsigned value, continuation;
    if (a >= 0xc2 && a <= 0xdf) { value = a & 31; continuation = 1; }
    else if (a >= 0xe0 && a <= 0xef) { value = a & 15; continuation = 2; }
    else if (a >= 0xf0 && a <= 0xf4) { value = a & 7; continuation = 3; }
    else return '?';
    for (unsigned i = 0; i < continuation; ++i) {
        if (p == end || (static_cast<unsigned char>(*p) & 0xc0) != 0x80) return '?';
        value = (value << 6) | (static_cast<unsigned char>(*p++) & 63);
    }
    if ((continuation == 1 && value < 128) || (continuation == 2 && value < 2048) ||
        (continuation == 3 && value < 65536) || value > 0x10ffff ||
        (value >= 0xd800 && value <= 0xdfff)) return '?';
    return value;
}
void glyph(Surface s, int x, int y, unsigned cp, int scale, uint16_t color) {
    unsigned base = cp;
    int mark = 0;
    switch (cp) {
    case 0xe1: base='a'; mark=1; break; case 0xe9: base='e'; mark=1; break;
    case 0xed: base='i'; mark=1; break; case 0xf3: base='o'; mark=1; break;
    case 0xfa: base='u'; mark=1; break; case 0xc1: base='A'; mark=1; break;
    case 0xc9: base='E'; mark=1; break; case 0xcd: base='I'; mark=1; break;
    case 0xd3: base='O'; mark=1; break; case 0xda: base='U'; mark=1; break;
    case 0xf1: base='n'; mark=2; break; case 0xd1: base='N'; mark=2; break;
    case 0xfc: base='u'; mark=3; break; case 0xdc: base='U'; mark=3; break;
    case 0xa1: base='!'; mark=4; break; case 0xbf: base='?'; mark=4; break;
    case 0x2018: case 0x2019: base='\''; break;
    case 0x201c: case 0x201d: base='"'; break;
    case 0x2013: case 0x2014: base='-'; break;
    case 0xa0: base=' '; break;
    }
    if (base < 32 || base > 126) base = '?';
    for (int row=0; row<14; ++row) {
        const unsigned bits = font8x14[base-32][mark == 4 ? 13-row : row];
        for (int col=0; col<8; ++col)
            if (bits & (1U << (7-col)))
                fill(s, {x+col*scale, y+(row+2)*scale, x+(col+1)*scale, y+(row+3)*scale}, color);
    }
    const unsigned marks[3][2] = {{0x08,0x10}, {0x34,0x58}, {0x28,0x28}};
    if (mark > 0 && mark < 4)
        for (int row=0; row<2; ++row)
            for (int col=0; col<8; ++col)
                if (marks[mark-1][row] & (1U << (7-col)))
                    fill(s, {x+col*scale,y+row*scale,x+(col+1)*scale,y+(row+1)*scale},color);
}
int layout(const unsigned *text, int count, int columns, int *starts, int *ends) {
    int at=0, lines=0;
    while (at<count && lines<256) {
        while (at<count && text[at]==' ') ++at;
        if (at == count) break;
        const int start=at;
        int last_space=-1;
        while (at<count && text[at]!='\n' && at-start<columns) {
            if (text[at]==' ') last_space=at;
            ++at;
        }
        int end=at;
        if (at<count && text[at]!='\n' && last_space>start) { end=last_space; at=last_space+1; }
        else if (at<count && text[at]=='\n') ++at;
        if (starts) starts[lines]=start;
        if (ends) ends[lines]=end;
        ++lines;
    }
    return lines;
}
void overlay(Surface s, Rect box, const char *utf8) {
    if (!utf8 || !*utf8) return;
    unsigned text[256];
    const char *p=utf8, *end=utf8;
    for (int i=0; i<1024 && *end; ++i) ++end;
    int count=0;
    while (p<end && count<256) text[count++]=next_character(p,end);
    if (!count) return;
    box.x0=std::clamp(box.x0-4,0,s.width); box.x1=std::clamp(box.x1+4,0,s.width);
    box.y0=std::clamp(box.y0-3,0,s.height); box.y1=std::clamp(box.y1+3,0,s.height);
    if (box.x1-box.x0<16 || box.y1<=box.y0) return;
    int scale=1, lines=0;
    for (int candidate=4; candidate>=1; --candidate) {
        int columns=std::max(1,(box.x1-box.x0-4)/(8*candidate));
        int required=layout(text,count,columns,nullptr,nullptr);
        if (required*17*candidate+4<=box.y1-box.y0 || candidate==1) {
            scale=candidate; lines=required; break;
        }
    }
    const int desired_height=std::max(box.y1-box.y0, std::min(s.height/2,lines*17*scale+4));
    box.y1=std::min(s.height,box.y0+desired_height);
    if (box.y1-box.y0<21) { box.y0=std::max(0,box.y1-21); }
    int starts[256], ends[256];
    lines=layout(text,count,std::max(1,(box.x1-box.x0-4)/(8*scale)),starts,ends);
    const int visible=std::min(lines,(box.y1-box.y0-4)/(17*scale));
    for (int line=0; line<visible; ++line) {
        int x=box.x0+2;
        for (int i=starts[line]; i<ends[line]; ++i,x+=8*scale) {
            unsigned cp=text[i];
            if (line==visible-1 && visible<lines && i>=ends[line]-3) cp='.';
            const int y=box.y0+2+line*17*scale;
            glyph(s,x-1,y,cp,scale,0);
            glyph(s,x+1,y,cp,scale,0);
            glyph(s,x,y-1,cp,scale,0);
            glyph(s,x,y+1,cp,scale,0);
            glyph(s,x,y,cp,scale,0xffff);
        }
    }
}
void hud(Surface s, const DisplayStats &stats) {
    char line[96];
    std::snprintf(line,sizeof(line),"%u.%u FPS | DET %u REC %u TR %u ms",stats.camera_fps_x10/10,
        stats.camera_fps_x10%10,stats.detector_ms,stats.recognizer_ms,stats.translator_ms);
    fill(s,{0,0,s.width,40},0);
    for (int i=0; line[i] && (i+1)*8<=s.width; ++i) glyph(s,4+i*8,1,line[i],1,0xffff);
    char utilization[24];
    if (stats.npu_active_percent>=0)
        std::snprintf(utilization,sizeof(utilization),"%d%% active",stats.npu_active_percent);
    else std::snprintf(utilization,sizeof(utilization),"active n/a");
    if (stats.cpu_ms && stats.npu_ms)
        std::snprintf(line,sizeof(line),"NPU %s | CPU/NPU %u.%ux",utilization,
            stats.cpu_ms/stats.npu_ms,(stats.cpu_ms%stats.npu_ms)*10/stats.npu_ms);
    else std::snprintf(line,sizeof(line),"NPU %s | speedup n/a",utilization);
    for (int i=0; line[i] && (i+1)*8<=s.width; ++i) glyph(s,4+i*8,21,line[i],1,0xffff);
}
}
Viewport camera_viewport(int sw,int sh,int ow,int oh) {
    if (sw<=0 || sh<=0 || ow<=0 || oh<=0) return {0,0,0,0};
    if (static_cast<int64_t>(sw)*oh>static_cast<int64_t>(sh)*ow) {
        int width=std::max(1,static_cast<int>(static_cast<int64_t>(sh)*ow/oh));
        return {(sw-width)/2,0,width,sh};
    }
    int height=std::max(1,static_cast<int>(static_cast<int64_t>(sw)*oh/ow));
    return {0,(sh-height)/2,sw,height};
}
bool render_translation(Surface out,Image image,const TranslationBox *boxes,size_t count,const DisplayStats *stats) {
    if (!out.pixels || !image.pixels || out.width<=0 || out.height<=0 || image.width<=0 ||
        image.height<=0 || out.stride<out.width || image.stride<image.width || (count && !boxes)) return false;
    const Viewport crop=camera_viewport(image.width,image.height,out.width,out.height);
    for (int y=0; y<out.height; ++y) {
        const uint16_t *row=image.pixels+(crop.y+y*crop.height/out.height)*image.stride;
        for (int x=0; x<out.width; ++x) out.pixels[y*out.stride+x]=row[crop.x+x*crop.width/out.width];
    }
    for (size_t i=0; i<std::min(count,size_t(32)); ++i) {
        const auto &b=boxes[i];
        if (!std::isfinite(b.x0) || !std::isfinite(b.y0) || !std::isfinite(b.x1) || !std::isfinite(b.y1) ||
            b.x1<=b.x0 || b.y1<=b.y0 || b.x1<=crop.x || b.y1<=crop.y ||
            b.x0>=crop.x+crop.width || b.y0>=crop.y+crop.height) continue;
        Rect r={static_cast<int>((std::clamp(b.x0,float(crop.x),float(crop.x+crop.width))-crop.x)*out.width/crop.width),
            static_cast<int>((std::clamp(b.y0,float(crop.y),float(crop.y+crop.height))-crop.y)*out.height/crop.height),
            static_cast<int>((std::clamp(b.x1,float(crop.x),float(crop.x+crop.width))-crop.x)*out.width/crop.width),
            static_cast<int>((std::clamp(b.y1,float(crop.y),float(crop.y+crop.height))-crop.y)*out.height/crop.height)};
        overlay(out,r,b.translated_utf8);
        if (b.outline_color) {
            fill(out,{r.x0,r.y0,r.x1,r.y0+3},b.outline_color);
            fill(out,{r.x0,r.y1-3,r.x1,r.y1},b.outline_color);
            fill(out,{r.x0,r.y0,r.x0+3,r.y1},b.outline_color);
            fill(out,{r.x1-3,r.y0,r.x1,r.y1},b.outline_color);
        }
    }
    if (stats) hud(out,*stats);
    return true;
}
}
