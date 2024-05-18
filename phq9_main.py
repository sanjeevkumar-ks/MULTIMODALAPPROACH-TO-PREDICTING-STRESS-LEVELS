from datetime import datetime
from time import strftime
from statistics import mean
import numpy as np
import pandas as pd
import csv
import os.path
#loading global variables
question_number = 1
appname = "MH Tracker"
now = datetime.now()
date = now.strftime("%Y-%m-%d")
time = now.strftime("%H:%M:%S")
start = True
#Read .csv into a DataFrame or create new savefile
if os.path.isfile('phq9.csv'):
    df = pd.read_csv('phq9.csv', header=0)
else:
    headers = ['date', 'time', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'total', 'severity']
    df = pd.DataFrame(columns = headers)
    df.to_csv('phq9.csv', index = False)
    df = pd.read_csv('phq9.csv', header=0)

###############################################################################################################    
def questionnaire():
    # Initialising variables.
    global start
    start = True
    global question_number
    score = []
    total = 0
    severity = ""
    choice = "\n1. Not at all\n2. Several Days\n3.More than half the days\n4. Nearly every day.\n"

    # check_ans checks if the score is in between 1 and 4, and that it is an integer
    def check_answer(ans):
        global question_number
        if ans.isdigit():
            ans = int(ans)
            if ans >= 1 and ans <= 4:
                score.append(ans)
                question_number += 1
            else:
                print("Input invalid. Please try again.")
                return ans
        else:
            print("Input invalid. Please try again.")
            return ans

    # Function to assess severity
    def assess_severity(x):
        if x >= 0 and x <= 4:
            return "subclinical"
        elif x >= 5 and x <= 9:
            return "mild"
        elif x >= 10 and x <= 14:
            return "moderate"
        elif x >= 15 and x <= 19:
            return "moderately severe"
        else:
            return "severe"

    # Introductory message
    print("This assessment can help you better understand your mental well-being over a long period of time by recording answers to your questions and saving this in a savefile. The savefile can be loaded again onto this program to keep track of your progress.\nIt is advised that you do NOT answer this questionnaire more than once per day.")

    while start == True:
        # User prompt to begin
        yn = input("Are you ready to take the questionnaire? Y/N\n").lower()
        if yn == "y":
            # PHQ-9 starts here if user selects "y"
            while question_number <= 10:
                if question_number == 1:
                    q1 = input("Over the last two weeks, have you had little interest or pleasure in doing things?%s" % choice)
                    check_answer(q1)
                elif question_number == 2:
                    q2 = input("Over the last two weeks, have you been feeling down, depressed, or hopeless?%s" % choice)
                    check_answer(q2)
                elif question_number == 3:
                    q3 = input("Over the last two weeks, have you had trouble falling asleep, staying asleep or sleeping too much?%s" % choice)
                    check_answer(q3)
                elif question_number == 4:
                    q4 = input("Over the last two weeks, have you been feeling tired or having little energy?%s" % choice)
                    check_answer(q4)
                elif question_number == 5:
                    q5 = input("Over the last two weeks, have you had poor appetite or were over-eating?:%s" % choice)
                    check_answer(q5)
                elif question_number == 6:
                    q6 = input("Over the last two weeks, have you been feeling bad about yourself - or that you are a failure or have let yourself or your family down?:%s" % choice)
                    check_answer(q6)
                elif question_number == 7:
                    q7 = input("Over the last two weeks, have you had trouble concentrating on things, such as reading the newspaper or watching television?:%s" % choice)
                    check_answer(q7)
                elif question_number == 8:
                    q8 = input("Over the last two weeks, have you moving or speaking so slowly that other people could have noticed? \n \
    Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual? %s" % choice)                
                    check_answer(q8)
                elif question_number == 9:
                    q9 = input("Over the last two weeks, have you had thoughts that you would be better off dead, or of hurting yourself in some way?: %s" % choice)
                    check_answer(q9)
                else:
                    total = sum(score) - 9
                    print("Thank you for answering these questions.")
                    print("The scores you have entered were " + str(score))
                    severity = assess_severity(total)
                    print("Your PHQ-9 score is " + str(total) +
                        ", which is classified by this questionnaire as " + severity + ".")
                    score.insert(0, date)
                    score.insert(1, time)
                    score.append(total)
                    score.append(severity)
                    print("Stress predicted Result =",severity)
                    #Code below writes data in a csv file
                    df.loc[len(df)] = score
                    df.to_csv('phq9.csv', index=False)

                    #Code above writes data in csv file
                    go_back = input(
                        "Press M to return to menu, X to exit program.").lower()
                    if go_back == "m":
                        main_screen()
                    elif go_back == "x":
                        start = False
                        break
                    else:
                        print("Please enter a valid input.")
        elif yn == "n":
            main_screen()
        else:
            print("Please enter a valid input.")
###################################################################################################################

def display_scores():
    global start
    while start == True:
        choice1 = input("Please enter how you would like your scores to be displayed.\nT = table\nG = Graph\nS = Summary\nR = Return\nD = Debug\nX = Exit\n").lower()
        if choice1 == "t":
            print(df)
        elif choice1 == "g":
            import matplotlib.pyplot as plt 
            #Placeholder message for graph depiction
            
            print("A graph should appear here.")
        elif choice1 == "s":
            print("This is the summary of recent PHQ-9 scores.")
            print(df[['date', 'time', 'total', 'severity']])
        elif choice1 == "r":
            main_screen()
        elif choice1 == "x":
            start = False
            break
        else:
            print("Please enter a valid input")
            


def main_screen():
    global start
    print("Today's date is %s" % date)
    print("This is the PHQ-9 questionnaire.\nPlease choose from the following options. \nPLEASE NOTE: This is NOT used as a diagnostic tool for depression and related disorders. Should there be genuine concerns about the mental well-being of you or someone you know, you should consult a medical professional for their assessment and/or treatment.")
    while start == True:
        choice = input(
            "Please choose from the following:\nA: Answer PHQ-9\nD: Display trend.\nX: Exit application.\n").lower()
        if choice == "a":
            questionnaire()
            start = False
        elif choice == "d":
            display_scores()
            start = False
        elif choice == "x":
            start = False
            break
        else:
            print("Please enter a valid input.")


main_screen()
