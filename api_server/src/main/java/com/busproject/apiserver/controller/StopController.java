package com.busproject.apiserver.controller;

import com.busproject.apiserver.entity.Stop; // 👈 Stop 엔티티 import 추가
import com.busproject.apiserver.service.StopService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import java.util.Map;

@RestController
@RequestMapping("/api/stops")
@CrossOrigin(origins = "*") // 👈 이 줄을 추가합니다.origins = "*"는 모든 출처(즉, 로컬 파일 시스템, 다른 도메인)에서의 접속을 허용한다는 의미입니다.
public class StopController {
    @Autowired
    private StopService stopService;

    // ---------------------------------------------------------------------------------------
    // 💡 수정된 부분: StopService의 getStopInfo를 호출하고, Stop 엔티티를 직접 반환하도록 변경합니다.
    //    Spring은 Stop 엔티티를 자동으로 JSON 형태로 변환해 줍니다.
    // ---------------------------------------------------------------------------------------
    @GetMapping("/{stopId}")
    // 반환 타입을 Map 대신 Stop 엔티티로 변경하여 자동 JSON 변환을 활용합니다.
    public Stop getStopInfo(@PathVariable String stopId) {

        // 💡 StopService의 getStopInfo 메소드를 호출합니다. (빨간불 해결)
        return stopService.getStopInfo(stopId);
    }
    // ---------------------------------------------------------------------------------------


    // YOLO가 호출할 주소: POST http://localhost:8080/api/stops/{stopId}/crowd
    @PostMapping("/{stopId}/crowd")
    public String updateCrowd(@PathVariable String stopId, @RequestBody Map<String, Integer> body) {
        int crowd = body.get("crowd");
        stopService.updateCrowd(stopId, crowd);

        System.out.println("✅ DB 업데이트 완료: " + stopId + " -> 혼잡도 " + crowd);
        return "Success";
    }
}