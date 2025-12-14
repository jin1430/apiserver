import cx_Oracle

try:
    # 유저이름: system (보통), 비밀번호: 아까 설정한 것 (예: 1234)
    dsn = cx_Oracle.makedsn('0.tcp.jp.ngrok.io', 17833, 'xe')
    connection = cx_Oracle.connect('bus_admin', '1234', dsn)

    print("접속 성공!")
    print("DB 버전:", connection.version)

    connection.close()

except Exception as e:
    print("접속 실패 ㅠㅠ 원인:", e)