// SPDX-License-Identifier: Apache-2.0
#include "camera.h"
#include <algorithm>
#include <cmath>

namespace dcops {
namespace {
bool valid(CameraPlanes image) {
    return image.pixels && image.width>0 && image.height>0 &&
        image.width<=8192 && image.height<=8192;
}
float channel(CameraPlanes image,int x,int y,int channel) {
    const int pixels=image.width*image.height;
    int index=y*image.width+x;
    if (image.rotated_180) index=pixels-1-index;
    return image.pixels[channel*pixels+index];
}
float sample(CameraPlanes image,float x,float y,int c) {
    x=std::clamp(x,0.f,float(image.width-1));
    y=std::clamp(y,0.f,float(image.height-1));
    const int x0=int(x),y0=int(y),x1=std::min(x0+1,image.width-1),y1=std::min(y0+1,image.height-1);
    const float fx=x-x0,fy=y-y0;
    const float a=channel(image,x0,y0,c)*(1-fx)+channel(image,x1,y0,c)*fx;
    const float b=channel(image,x0,y1,c)*(1-fx)+channel(image,x1,y1,c)*fx;
    return a*(1-fy)+b*fy;
}
}
bool detector_input(CameraPlanes image,float *out,size_t elements,int width,int height) {
    if (!valid(image) || !out || width<=0 || height<=0 || width>288 || height>480 ||
        elements<size_t(3*width*height)) return false;
    const auto crop=camera_viewport(image.width,image.height,width,height);
    const float mean[3]={.485f,.456f,.406f},stddev[3]={.229f,.224f,.225f};
    for (int y=0;y<height;++y) for (int x=0;x<width;++x) {
        const float sx=crop.x+(x+.5f)*crop.width/width-.5f;
        const float sy=crop.y+(y+.5f)*crop.height/height-.5f;
        for (int c=0;c<3;++c)
            out[c*width*height+y*width+x]=(sample(image,sx,sy,2-c)/255.f-mean[c])/stddev[c];
    }
    return true;
}
int recognizer_input(CameraPlanes image,Viewport region,float *out,size_t elements) {
    constexpr int width=320,height=48;
    if (!valid(image) || !out || elements<size_t(3*width*height) || region.x<0 || region.y<0 ||
        region.width<=0 || region.height<=0 || region.x>image.width-region.width ||
        region.y>image.height-region.height) return 0;
    const int resized_width=std::min(width,static_cast<int>((int64_t(height)*region.width+region.height-1)/region.height));
    std::fill(out,out+3*width*height,0.f);
    for (int y=0;y<height;++y) for (int x=0;x<resized_width;++x) {
        const float sx=std::clamp(region.x+(x+.5f)*region.width/resized_width-.5f,
                                 float(region.x),float(region.x+region.width-1));
        const float sy=std::clamp(region.y+(y+.5f)*region.height/height-.5f,
                                 float(region.y),float(region.y+region.height-1));
        for (int c=0;c<3;++c)
            out[c*width*height+y*width+x]=sample(image,sx,sy,2-c)/127.5f-1.f;
    }
    return resized_width;
}
TranslationBox detector_box_to_source(CameraPlanes image,float x0,float y0,float x1,float y1,const char *translation,
                                     int width,int height) {
    if (!valid(image) || width<=0 || height<=0 || width>288 || height>480 ||
        !std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) || !std::isfinite(y1)) return {};
    const auto crop=camera_viewport(image.width,image.height,width,height);
    return {crop.x+std::clamp(x0,0.f,float(width))*crop.width/width,
        crop.y+std::clamp(y0,0.f,float(height))*crop.height/height,
        crop.x+std::clamp(x1,0.f,float(width))*crop.width/width,
        crop.y+std::clamp(y1,0.f,float(height))*crop.height/height,translation};
}
}
