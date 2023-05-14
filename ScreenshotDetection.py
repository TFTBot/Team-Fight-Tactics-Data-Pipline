import cv2 as cv
from ultralytics import YOLO
from cap_from_youtube import cap_from_youtube as yt
import numpy as np
from difflib import SequenceMatcher
import os
import pickle

#create windows for everything that is tracked
cv.namedWindow("video", cv.WINDOW_NORMAL)

ytList = pickle.load(open("mortdog_videos.pkl","rb"))

model = YOLO("./runs/detect/yolov8s_phase2/weights/best.pt")

#initalize CombatCounter
combatCount = 0

for video in ytList: 

    url = "https://www.youtube.com/watch?v=" + video
    try:
        cap = yt(url, "1080p60")
    except ValueError:
        cap = yt(url,"720p")
    
    
    #frame counter
    cntr = 0
    
    #initialize Variables
    isFirstZero = False
    firstZeroFlag = False
    firstPlanningFlag = False
    neutralMobsFlag = False
    firstZeroCounter = 0
    
#video loop
    while True:
        ret, src = cap.read()
        
        if not ret:
            print ("no video input")
            break
    
        
        frame = cv.resize(src, (1280,720), interpolation = cv.INTER_AREA)
        cntr += 1
        

    # check every nth frame
        if ((cntr%30) == 0):
            
            resultList = model (frame)
            result = resultList.__getitem__(0)
            positions = result.boxes.data.cpu().numpy()
                
            if (4 in result.boxes.cls.cpu()): #check if we see a zero
                if (firstZeroFlag == False): #check if it is the first zero
                    for position in positions:
                        if (position[5] == 4):
                            if (position[4] > 0.7): #check if the confidence of the Zero is greater than 0.5
                                isFirstZero = True
                                firstZeroFlag = True
                                firstZeroCounter += 1
                                print (firstZeroCounter)
                                print (position[4])
                                break
                    if (isFirstZero and firstZeroCounter == 2): #if it's the second Zero we see, save the screenshot
                        combatCount += 1
                        os.chdir("./Precombat")
                        cv.imwrite("PrecombatScreenshot%d.jpeg" % combatCount, frame)
                        print ()
                        os.chdir("../")
                    elif (isFirstZero and firstZeroCounter == 8): #if it's the 8th Zero we see, no planning phase happened after combat
                        neutralMobsFlag = True
                        
                        recentPosition = 0
                        recentGame = "unknown"
                        for position in positions:
                            if (position[5] == 1 or position[5] == 3): #check for win & lose positions
                                if (position[0] > recentPosition): #take the right most result as the last game
                                    recentPosition = position[0]
                                    if (position[5] == 3):
                                        recentGame = "Win"
                                    elif (position[5] == 1):
                                        recentGame = "Lose"
                        if(recentGame != "unknown"):
                            os.chdir("./Postcombat")
                            cv.imwrite(f"PostcombatScreenshot{combatCount}{recentGame}.jpeg", frame)
                            os.chdir("../")
                else:
                    isFirstZero = False
            else:
                firstZeroFlag = False

            #check if in combat or planning phase
            if (0 in result.boxes.cls.cpu()): #combat
                firstPlanningFlag = False
                neutralMobsFlag = False
            elif (2 in result.boxes.cls.cpu()): #planning
                firstZeroCounter = 0
                if (firstPlanningFlag == False and neutralMobsFlag == False): #check for first frame of the planning phase
                    firstPlanningFlag = True
                    
                    recentPosition = 0
                    recentGame = "unknown"
                    for position in positions:
                        if (position[5] == 1 or position[5] == 3):#check for win & lose positions
                            if (position[0] > recentPosition): #take the right most result as the last game
                                recentPosition = position[0]
                                if (position[5] == 3):
                                    recentGame = "Win"
                                elif (position[5] == 1):
                                    recentGame = "Lose"
                    
                    if (recentGame != "unknown"):
                        os.chdir("./Postcombat")
                        cv.imwrite(f"PostcombatScreenshot{combatCount}{recentGame}.jpeg", frame)
                        os.chdir("../")

        cv.imshow ("video", frame) 
    
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
        
    cap.release()
    cv.destroyAllWindows()