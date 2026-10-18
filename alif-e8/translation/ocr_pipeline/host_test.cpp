#include "postprocess.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

unsigned token(char c) {
    for (unsigned i = 1; i < 97; ++i) if (dcops::kOcrCharacters[i] == c) return i;
    assert(false); return 0;
}
int main() {
    dcops::DetectionWorkspace workspace;
    std::vector<float> map(288 * 480, 0);
    auto fill = [&](unsigned x0, unsigned y0, unsigned w, unsigned h, float p) {
        for (unsigned y = y0; y < y0 + h; ++y) for (unsigned x = x0; x < x0 + w; ++x) map[y * 288 + x] = p;
    };
    fill(0, 0, 30, 10, .9f); fill(100, 200, 60, 15, .8f);
    fill(250, 450, 38, 30, .95f); fill(100, 100, 30, 10, .4f);
    map[1000] = std::numeric_limits<float>::quiet_NaN();
    dcops::DetectedBox boxes[4]; size_t count = 0;
    assert(dcops::detect_boxes(map.data(), 288, 480, boxes, 4, count, workspace) == dcops::OcrResult::Ok);
    assert(count == 3);
    for (size_t i = 0; i < count; ++i) {
        assert(boxes[i].x >= 0 && boxes[i].y >= 0 && boxes[i].w > 0 && boxes[i].h > 0);
        assert(boxes[i].x + boxes[i].w <= 288 && boxes[i].y + boxes[i].h <= 480);
        if (i) assert(boxes[i-1].w * boxes[i-1].h >= boxes[i].w * boxes[i].h);
    }
    assert(dcops::detect_boxes(map.data(), 288, 480, boxes, 1, count, workspace) == dcops::OcrResult::Ok && count == 1);
    assert(boxes[0].x > 230 && boxes[0].y > 430);
    std::fill(map.begin(), map.end(), 0);
    // Four disconnected characters share one baseline; crop the complete word.
    for (unsigned x : {10u, 22u, 34u, 46u}) fill(x, 50, 8, 15, .9f);
    for (unsigned x : {10u, 22u}) fill(x, 90, 8, 15, .9f);
    fill(200, 50, 8, 15, .9f);
    assert(dcops::detect_boxes(map.data(), 288, 480, boxes, 4, count, workspace) == dcops::OcrResult::Ok && count == 3);
    assert(boxes[0].x <= 10 && boxes[0].x + boxes[0].w >= 54);
    assert(boxes[0].y < 50 && boxes[0].y + boxes[0].h < 90);
    assert(boxes[1].y > 70 && boxes[2].x > 180);
    std::fill(map.begin(), map.end(), 1);
    assert(dcops::detect_boxes(map.data(), 288, 480, boxes, 4, count, workspace) == dcops::OcrResult::Ok && count == 1);
    assert(boxes[0].x == 0 && boxes[0].y == 0 && boxes[0].w == 288 && boxes[0].h == 480);
    std::fill(map.begin(), map.end(), 0);
    assert(dcops::detect_boxes(map.data(), 288, 480, boxes, 4, count, workspace) == dcops::OcrResult::Ok && count == 0);
    assert(dcops::detect_boxes(map.data(), 289, 480, boxes, 4, count, workspace) == dcops::OcrResult::InvalidInput);
    std::vector<float> logits(40 * 97, -10);
    for (unsigned t = 0; t < 40; ++t) logits[t * 97] = 10;
    unsigned path[] = {token('H'), token('H'), 0, token('e'), token('l'), 0, token('l'), token('o')};
    for (unsigned t = 0; t < 8; ++t) { logits[t * 97] = -10; logits[t * 97 + path[t]] = 10; }
    logits[39 * 97] = -10; logits[39 * 97 + token('-')] = 10;
    char text[256]; float confidence;
    assert(dcops::decodeCTC(logits.data(), 40, 97, 64, text, sizeof(text), &confidence) == dcops::OcrResult::Ok);
    assert(std::strcmp(text, "Hello") == 0 && confidence > .99f);
    assert(dcops::decodeCTC(logits.data(), 40, 97, 320, text, sizeof(text), nullptr) == dcops::OcrResult::Ok);
    assert(std::strcmp(text, "Hello-") == 0);
    assert(dcops::decodeCTC(logits.data(), 40, 97, 64, text, 3) == dcops::OcrResult::OutputTooSmall && text[0] == 0);
    assert(dcops::decodeCTC(logits.data(), 40, 96, 64, text, sizeof(text)) == dcops::OcrResult::InvalidInput);
    logits[0] = std::numeric_limits<float>::infinity();
    assert(dcops::decodeCTC(logits.data(), 40, 97, 64, text, sizeof(text)) == dcops::OcrResult::InvalidInput && text[0] == 0);
    dcops::TextStability track;
    dcops::DetectedBox a{10,20,80,15,.9f}, jitter{12,21,80,15,.9f}, moved{200,20,80,15,.9f};
    // Two readings alone must not freeze the camera before the one-second hold.
    assert(!track.observe("Hello", a, 100) && !track.confirmed());
    assert(!track.observe("Hello", jitter, 600) && track.matches() == 2 && !track.confirmed());
    assert(track.observe("Hello", a, 1100) && track.confirmed() && track.stableElapsed() == 1000);
    assert(!track.cachedFor("Hello"));
    assert(track.setTranslation("Hola", 1150) && track.visible(1200));
    // Unchanged source reuses successful output without another translation request.
    assert(!track.observe("Hello", jitter, 1300) && track.confirmed() && track.cachedFor("Hello"));
    assert(std::strcmp(track.translation(), "Hola") == 0 && track.box().x == jitter.x);
    // Same text at another ROI needs a fresh hold, then reuses cached Spanish at that ROI.
    assert(!track.observe("Hello", moved, 1500) && !track.confirmed() && track.matches() == 1);
    assert(!track.observe("Hello", moved, 2500) && track.confirmed() && track.cachedFor("Hello"));
    assert(track.box().x == moved.x);
    // An empty read or changed text breaks consecutive stability without deleting the cache.
    assert(!track.observe("", {}, 2600) && track.matches() == 0 && !track.confirmed());
    assert(!track.observe("Hello", a, 3000));
    assert(!track.observe("Help", a, 3500) && track.matches() == 1);
    assert(!track.observe("Help", a, 4000) && !track.confirmed());
    assert(track.observe("Help", a, 4500) && track.confirmed() && !track.cachedFor("Help"));
    assert(!track.setTranslation("", 4510));  // Failure/empty output never enters cache.
    assert(track.cachedFor("Hello") && std::strcmp(track.translation(), "Hola") == 0);
    assert(track.setTranslation("Ayuda", 4600));
    assert(track.cachedFor("Help") && !track.cachedFor("Hello"));
    assert(!track.visible(10000));
    // Gradual motion is checked against the first ROI, not only its last position.
    track.reset();
    assert(!track.observe("Hello", a, 100));
    assert(!track.observe("Hello", {30,20,80,15,.9f}, 600));
    assert(!track.observe("Hello", {50,20,80,15,.9f}, 1100) && track.matches() == 1 && !track.confirmed());
    // A long observation gap resets the hold; unsigned timestamps handle rollover.
    assert(!track.observe("Hello", {50,20,80,15,.9f}, 5000) && track.matches() == 1);
    track.reset();
    assert(!track.observe("X", a, 0xfffffff0u));
    assert(!track.observe("X", a, 0x20) && !track.confirmed());
    assert(track.observe("X", a, 0x3d8) && track.confirmed());
    assert(track.setTranslation("Y", 0x3e0) && track.visible(0x400));
    std::cout << "{\"geometry_cases\":9,\"ctc_cases\":5,\"tracking_cases\":16,\"workspace_bytes\":" << sizeof(workspace) << "}\n";
}
