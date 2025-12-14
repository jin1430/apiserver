package com.busproject.apiserver.service;


import com.busproject.apiserver.entity.Stop;
import com.busproject.apiserver.repository.StopRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class StopService {
    @Autowired
    private StopRepository stopRepository;

    @Transactional
    public void updateCrowd(String stopId, int crowdLevel) {
        Stop stop = stopRepository.findById(stopId)
                .orElseThrow(() -> new RuntimeException("정류장 없음: " + stopId));
        stop.setCrowd(crowdLevel);
        // @Transactional 덕분에 save를 호출하지 않아도 자동으로 UPDATE 쿼리가 나갑니다.
    }
    // -----------------------------------------------------------
    // 💡 추가해야 할 메소드: DB에서 Stop 엔티티를 조회하여 반환
    // -----------------------------------------------------------
    @Transactional(readOnly = true) // 읽기 전용 트랜잭션 설정
    public Stop getStopInfo(String stopId) {
        return stopRepository.findById(stopId)
                .orElseThrow(() -> new RuntimeException("정류장 없음: " + stopId));
    }
}