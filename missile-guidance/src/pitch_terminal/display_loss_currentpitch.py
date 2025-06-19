# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:43:42 2024

@author: ammar
"""

from src.receiver import readPacket
from src.utils import unpause_game, switch_tab
import socket, time


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("localhost", 2873))

switch_tab()
time.sleep(1)
unpause_game()



grounded = False
tar_final_pitch = -30

while not grounded:
    try:
        
        d, a = sock.recvfrom(2048)
        values = readPacket(d)          
            
        grounded = values.get('grounded') or values.get('destroyed')
        current_LOS = values.get('new_pitch')
        current_pitch = values.get('pitch')
        
        x = values.get('pos_x')
        y = values.get('pos_y')
        z = values.get('pos_z')
        target_pos_x = values.get('target_pos_x')
        target_pos_y = values.get('target_pos_y')
        target_pos_z = values.get('target_pos_z')
        
        
        
        target_min_LOS = tar_final_pitch - current_LOS
        LOS_min_current = current_LOS - current_pitch
        
#         text_data = f"""----- flight_data -----
# target final pitch:  {"{:.3f}".format(tar_final_pitch)}
# current LOS:         {"{:.3f}".format(current_LOS)}
# target - LOS:        {"{:.3f}".format(target_min_LOS)}
# current pitch:       {"{:.3f}".format(current_pitch)}
# LOS - pitch:         {"{:.3f}".format(LOS_min_current)}
# -----------------------

# """

        text_data = f"""----- flight_data -----
target final pitch:  {"{:.3f}".format(tar_final_pitch)}
current LOS:         {"{:.3f}".format(current_LOS)}
target - LOS:        {"{:.3f}".format(target_min_LOS)}
current pitch:       {"{:.3f}".format(current_pitch)}
LOS - pitch:         {"{:.3f}".format(LOS_min_current)}
-----------------------

"""
        print(text_data)
        
    except socket.timeout:
        print('\n[ERROR]: TimeoutError at get_observation method\n')

