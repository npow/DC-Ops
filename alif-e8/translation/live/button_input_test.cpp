// SPDX-License-Identifier: Apache-2.0
#include "button_input.h"
#include <cassert>
#include <cstdint>
#include <initializer_list>
int main() {
    for (uint32_t pin=1;pin<=16;pin<<=1) {
        dcops::ButtonInput button;
        assert(button.press(pin,pin,0));
        assert(!button.press(pin,0,150));
        assert(!button.press(pin,0,300));
        assert(!button.press(0,0,310));
        assert(button.press(pin,pin,500));
    }
    {
        dcops::ButtonInput button;
        assert(button.press(4,4,0));
        assert(!button.press(4,4,100));
        assert(!button.press(4,4,300));
        assert(!button.press(4,4,1000));
        assert(!button.press(0,4,1100)); // Stale level at release is not another press.
        assert(button.press(4,4,1300));
    }
    for (uint32_t pin : {1u,2u,8u,16u}) {
        dcops::ButtonInput button;
        assert(button.press(0,pin,0)); // Short tap completed between polls.
        assert(!button.press(0,0,300));
        assert(button.press(pin,pin,500));
        assert(button.press(pin,pin,700)); // Release/repress missed during processing.
        assert(!button.press(pin,0,1000));
    }
    {
        dcops::ButtonInput button;
        assert(button.press(0,4,0)); // A sampled pin2 tap while no held level was observed.
        assert(!button.press(0,0,300));
    }
    {
        dcops::ButtonInput button;
        assert(button.press(8,8,0));
        assert(!button.press(0,0,5));
        assert(!button.press(8,8,10));
        assert(!button.press(0,8,20));
        assert(!button.press(8,8,149));
        assert(!button.press(8,0,300)); // Debounced bounce must not become a delayed press.
        assert(button.press(0,8,450));
    }
    {
        dcops::ButtonInput button;
        assert(!button.press(0xffffffe0,0xffffffe0,0));
        assert(button.press(16,16,0));
        assert(!button.press(0,0,1));
        assert(!button.press(1,1,149));
        assert(button.press(2,2,150));
    }
    {
        dcops::ButtonInput button;
        assert(button.press(1,1,UINT32_MAX-100));
        assert(!button.press(0,0,UINT32_MAX-50));
        assert(!button.press(1,1,20));
        assert(button.press(1,1,60)); // Unsigned elapsed time crosses timer wrap.
    }
}
