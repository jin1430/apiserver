package com.busproject.apiserver.entity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "STOPS")
public class Stop {
    @Id
    @Column(name = "stop_id")
    private String stopId;

    @Column(name = "stop_name")
    private String stopName;

    @Column(name = "route_order")
    private Integer routeOrder;

    @Column(name = "crowd")
    private Integer crowd;

    // Getters and Setters
    public String getStopId() { return stopId; }
    public void setStopId(String stopId) { this.stopId = stopId; }
    public Integer getCrowd() { return crowd; }
    public void setCrowd(Integer crowd) { this.crowd = crowd; }
    // 나머지 Getter/Setter는 필요하면 추가
}