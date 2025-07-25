LOCAL_FOLDER = "data/"

TICKER_DATA_RAW_FILENAME_PREFIX = "single_raw_"
TICKER_DATA_W_FEATURES_FILENAME_PREFIX = "single_with_features_"
DATA_FILES_EXTENSION = ".xlsx"

TRADE_ALREADY_HALF_CLOSED = "; partially_closed"
CLOSED_VOLATILITY_SPIKE = "; closed_due to volatility spike"
CLOSED_SIDE_CHANGE = "; closed_due to forecast side change"
CLOSED_HAMMER = "; closed_due to hammer candle"
CLOSED_SHOOTING_STAR = "; closed_due to shooting star candle"
CLOSED_MAX_DURATION = "; closed because max duration exceeded"
SL_TRIGGERED = "; stop-loss triggered"
TP_TRIGGERED = "; take profit triggered"
SL_TIGHTENED = "; stop-loss tightened during volatility spike"

# SS - special situation
SS_NO_TODAY = "No special situation today"
SS_PARTIAL_CLOSE = "SS Partial close"
SS_HAMMER = "SS Hammer"
SS_SHOOTING_STAR = "SS Shooting star"
SS_VOLATILITY_SPIKE = "SS Volatility spike"
SS_OVERBOUGHT_OVERSOLD = "SS Overbought / Oversold"
SS_MAX_DURATION = "SS Max trade duration exceeded"

# DPS - desired position size
DPS_NO_START_HIGH_VOLATILITY = "DPS Don't start new position when high ATR"
DPS_NO_ADD_HIGH_VOLATILITY = "DPS Don't add to position when high ATR"
DPS_NO_START_HIGH_MTUM = "DPS Don't start new position when high momentum"
DPS_WAIT_SL_TRIGGERED_TODAY = "DPS last_trade stop loss triggered today, wait"
DPS_WAIT_VOLATILITY_SPIKE = "DPS Last trade was closed volatility spike, wait"
DPS_WAIT_SL_SAME_SIDE = "DPS Last trade closed SL. To enter the same side, wait"
DPS_WAIT_SHOOTING_STAR = "DPS Last trade was closed shooting star, wait to get long"
DPS_WAIT_HAMMER = "DPS Last trade was closed hammer, wait to get short"
DPS_TAKE_BET = "DPS risk/reward is good, take the bet"
DPS_BAD_RR_PASS = "DPS risk/reward is bad, pass"
DPS_BAD_RR_NO_ADD = "DPS risk/reward is bad, don't add to trades"
DPS_NO_CLOSE_LOSERS = "DPS no partial close of loosing trades"
DPS_ADD_WINNER_YES = "DPS add to winner in good R/R - YES"
DPS_ADD_WINNER_NO = "DPS add to winner in good R/R - NO"
DPS_MAIN_FLOW_YES = "DPS Main flow - yes"
DPS_MAIN_FLOW_NO = "DPS Main flow - no"
DPS_STUB = "DPS get_desired_current_position_size stub"

tickers_all = ["J203.JO", "ABG.JO", "ADH.JO", "AEL.JO", "AFE.JO", "AFH.JO", "AFT.JO", "AGL.JO", "AHR.JO", "AIP.JO", "ANG.JO", "ANH.JO", "APN.JO", "ARI.JO",
          "ARL.JO", "ATT.JO", "AVI.JO", "BAW.JO", "BHG.JO", "BID.JO", "BLU.JO", "BOX.JO", "BTI.JO", "BTN.JO", "BVT.JO", "BYI.JO", "CFR.JO", "CLS.JO",
          "CML.JO", "COH.JO", "CPI.JO", "CSB.JO", "DCP.JO", "DRD.JO", "DSY.JO", "DTC.JO", "EMI.JO", "EQU.JO", "EXX.JO", "FBR.JO", "FFB.JO", "FSR.JO",
          "FTB.JO", "GFI.JO", "GLN.JO", "GND.JO", "GRT.JO", "HAR.JO", "HCI.JO", "HDC.JO", "HMN.JO", "HYP.JO", "IMP.JO", "INL.JO", "INP.JO", "ITE.JO",
          "JSE.JO", "KAP.JO", "KIO.JO", "KRO.JO", "KST.JO", "LHC.JO", "LTE.JO", "MCG.JO", "MKR.JO", "MNP.JO", "MRP.JO", "MSP.JO", "MTH.JO", "MTM.JO",
          "MTN.JO", "N91.JO", "NED.JO", "NPH.JO", "NPN.JO", "NRP.JO", "NTC.JO", "NY1.JO", "OCE.JO", "OMN.JO", "OMU.JO", "OUT.JO", "PAN.JO", "PHP.JO",
          "PIK.JO", "PMR.JO", "PPC.JO", "PPH.JO", "PRX.JO", "QLT.JO", "RBX.JO", "RCL.JO", "RDF.JO", "REM.JO", "RES.JO", "RLO.JO", "RNI.JO", "S32.JO",
          "SAC.JO", "SAP.JO", "SBK.JO", "SHC.JO", "SHP.JO", "SLM.JO", "SNT.JO", "SNT.JO", "SOL.JO", "SPG.JO", "SPP.JO", "SRE.JO", "SRI.JO", "SSS.JO",
          "SSU.JO", "SSW.JO", "SUI.JO", "TBS.JO", "TFG.JO", "TGA.JO", "TKG.JO", "TRU.JO", "TSG.JO", "VAL.JO", "VKE.JO", "VOD.JO", "WBC.JO", "WHL.JO"]

tickers_precious = [
    "GLD",
    "SLV",
    "GDX",
]

tickers_commodities = [
    "USO",
    "WEAT",
    "SOYB",
    "CORN",
]

tickers_stocks = [
    "SPY",
    "ARKK",
    "QQQ",
    "XME",
]

LOG_FILE = "app_run.log"

ACTION_BUY = "Buy"
ACTION_SELL = "Sell"
ACTION_DO_NOTHING = "Do nothing"
ACTION_CLOSE_POSITION = "Close position"
ACTION_SHARE_COUNT_0 = "Shares count 0"

DEFAULT_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

NUM_DAYS_FWD_RETURN = 4

FEATURE_COL_NAME_BASIC = "feature_basic"
FEATURE_COL_NAME_ADVANCED = "feature_advanced"