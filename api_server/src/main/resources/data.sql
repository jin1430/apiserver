-- 1. 정류장 데이터 삽입
-- (stop_id, stop_name, route_order, crowd)
INSERT INTO stops VALUES ('baekseok', '백석대학교', 1, 1);
INSERT INTO stops VALUES ('terminal', '천안터미널', 2, 1);
INSERT INTO stops VALUES ('cheonan_stn', '천안역', 3, 1);
INSERT INTO stops VALUES ('dujeong_stn', '두정역', 4, 1);


-- 2. 셔틀 위치 데이터 삽입
-- (vehicle_number, direction, current_stop_id, arrival_minutes)
INSERT INTO vehicles VALUES ('71라3156', 'outbound', 'baekseok', 1);
INSERT INTO vehicles VALUES ('75마1635', 'outbound', 'terminal', 5);
INSERT INTO vehicles VALUES ('71서5468', 'outbound', 'cheonan_stn', 0);
INSERT INTO vehicles VALUES ('78다3549', 'outbound', NULL, -1); -- 'return'은 특정 정류장이 아니므로 NULL