// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <stdint.h>

namespace dcops {
class ButtonInput {
public:
    bool press(uint32_t pins_down,uint32_t latched_events,uint32_t now_ms) {
        constexpr uint32_t mask=0x1f,edge_pins=0x1b,level_pin=0x04;
        const uint32_t down=pins_down&mask;
        const uint32_t events=latched_events&mask;
        const uint32_t newly_down=down&~previous_down_;
        // Pin2 is level-sensitive: a repeatedly reported held level is not a new tap.
        const uint32_t level_tap=events&level_pin&~down&~previous_down_;
        previous_down_=down;
        if (!(newly_down|(events&edge_pins)|level_tap)) return false;
        if (have_press_ && uint32_t(now_ms-last_press_ms_)<150) return false;
        have_press_=true;
        last_press_ms_=now_ms;
        return true;
    }
private:
    uint32_t previous_down_=0,last_press_ms_=0;
    bool have_press_=false;
};
}
