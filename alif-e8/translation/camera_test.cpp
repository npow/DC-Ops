// SPDX-License-Identifier: Apache-2.0
#include "camera.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
int main() {
    constexpr int w=640,h=480,n=w*h;
    std::vector<uint8_t> planes(n*3);
    std::fill(planes.begin(),planes.begin()+n,255);
    std::fill(planes.begin()+n,planes.begin()+2*n,127);
    dcops::CameraPlanes image{planes.data(),w,h,true};
    auto box=dcops::detector_box_to_source(image,0,0,288,480,"actual result");
    assert(box.x0==176 && box.y0==0 && box.x1==464 && box.y1==480);
    std::vector<float> detector(3*480*288+2,12345.f);
    assert(dcops::detector_input(image,detector.data()+1,detector.size()-2));
    assert(std::abs(detector[1]-(0.f-.485f)/.229f)<1e-5);
    assert(std::abs(detector[1+480*288]-(127.f/255-.456f)/.224f)<1e-5);
    assert(std::abs(detector[1+480*288*2]-(1.f-.406f)/.225f)<1e-5);
    assert(detector.front()==12345.f && detector.back()==12345.f);
    std::vector<float> recognition(3*48*320+2,12345.f);
    int resized=dcops::recognizer_input(image,{176,100,100,50},recognition.data()+1,recognition.size()-2);
    assert(resized==96);
    for (int c=0;c<3;++c) for (int y=0;y<48;++y) for (int x=96;x<320;++x)
        assert(recognition[1+c*48*320+y*320+x]==0.f);
    assert(recognition[1]==-1.f && recognition[1+2*48*320]==1.f);
    assert(recognition.front()==12345.f && recognition.back()==12345.f);
    // A bright physical bottom-right pixel becomes upright top-left under 180-degree correction.
    std::fill(planes.begin(),planes.end(),0);
    planes[n*3-1]=255;
    resized=dcops::recognizer_input(image,{0,0,1,1},recognition.data()+1,recognition.size()-2);
    assert(resized==48 && recognition[1]==1.f);
    image.rotated_180=false;
    resized=dcops::recognizer_input(image,{0,0,1,1},recognition.data()+1,recognition.size()-2);
    assert(resized==48 && recognition[1]==-1.f);
    assert(!dcops::detector_input(image,detector.data(),4));
    assert(!dcops::recognizer_input(image,{-1,0,5,5},recognition.data(),recognition.size()));
    assert(!dcops::recognizer_input(image,{0,0,100000,5},recognition.data(),recognition.size()));
    image.width=320; image.height=240;
    box=dcops::detector_box_to_source(image,0,0,288,480,nullptr);
    assert(box.x0==88 && box.x1==232 && box.y0==0 && box.y1==240);
}
