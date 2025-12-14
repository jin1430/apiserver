package com.busproject.apiserver.repository;

import com.busproject.apiserver.entity.Stop;
import org.springframework.data.jpa.repository.JpaRepository;

public interface StopRepository extends JpaRepository<Stop, String> {
    // 기본 기능(findById, save 등)이 자동으로 제공됩니다.
}