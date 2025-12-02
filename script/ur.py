import rtde_receive
rtde_r = rtde_receive.RTDEReceiveInterface("172.16.0.7")
actual_q = rtde_r.getActualQ()
print(actual_q)


