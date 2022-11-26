from random import randint

def getRandomYear():
    r = randint(0,46)
    if r < 4:
        return 1993
    if r < 18:
        return 1994
    if r < 30:
        return 1995
    return 1996

def getRandomDay(month):
    if month in [1,3,5,7,8,10,12]:
        return randint(0,31)
    if month == 2:
        return randint(0,28)
    return randint(0,30)

def getRandomDuration():
    r = randint(0,46)
    if r < 7:
        return 12
    if r < 18:
        return 24
    if r < 29:
        return 36
    if r < 38:
        return 48
    return 60

def getRandomFrequency():
    r = randint(0,4500)
    if r < 4167:
        return "monthly issuance"
    if r < 4407:
        return "weekly issuance"
    return "issuance after transaction"

def getNormalizedAverageTransAmount(avg_trans_am):
    answer =  (avg_trans_am-200) / 28563
    if answer > 1 or answer < 0:
        print("VALUE OUT OF BOUNDS" + str(answer))
    return answer

def getNormalizedAverageBalance(avg_bal):
    answer =  (avg_bal-200) / 82300
    if answer > 1 or answer < 0:
        print("VALUE OUT OF BOUNDS" + str(answer))
    return answer

def getNormalizedMinBalance(min_bal):
    answer =  (min_bal+13588) / 45639
    if answer > 1 or answer < 0:
        print("VALUE OUT OF BOUNDS" + str(answer))
    return answer

def getNormalizedMaxBalance(max_bal):
    answer =  (max_bal-200) / 193710
    if answer > 1 or answer < 0:
        print("VALUE OUT OF BOUNDS" + str(answer))
    return answer

def getNormalizedMinTransAmount(min_trans_am):
    answer =  (min_trans_am) / 1100
    if answer > 1 or answer < 0:
        print("VALUE OUT OF BOUNDS" + str(answer))
    return answer

def getNormalizedMaxTransAmount(max_trans_am):
    answer =  (max_trans_am-200) / 86200
    if answer > 1 or answer < 0:
        print("VALUE OUT OF BOUNDS" + str(answer))
    return answer

def getNormalizedLoanAmount(am):
    answer =  (am-4980) / 533520
    if answer > 1 or answer < 0:
        print("VALUE OUT OF BOUNDS" + str(answer))
    return answer

def getDenormalizedAverageTransAmount(avg_trans_am):
    return avg_trans_am * 28563 + 200

def getDenormalizedAverageBalance(avg_bal):
    return avg_bal * 82300 + 200

def getDenormalizedMinBalance(min_bal):
    return min_bal * 45639 - 13588

def getDenormalizedMaxBalance(max_bal):
    return max_bal * 193710 + 200

def getDenormalizedMinTransAmount(min_trans_am):
    return min_trans_am * 1100

def getDenormalizedMaxTransAmount(max_trans_am):
    return max_trans_am * 86200 + 200

def getDenormalizedLoanAmount(loan_am):
    return loan_am * 533520 + 4980







