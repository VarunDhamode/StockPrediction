import streamlit as st
sp500_stocks = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "FB", "TSLA", "BRK.B", "NVDA", "JPM",
    "JNJ", "V", "MA", "PYPL", "DIS", "ADBE", "HD", "BAC", "CMCSA", "XOM",
    "UNH", "INTC", "VZ", "NFLX", "CRM", "KO", "T", "PEP", "MRK", "ABBV",
    "PFE", "WMT", "NKE", "CVX", "TMO", "CSCO", "MDT", "ORCL", "AMGN", "ABT",
    "BMY", "ACN", "LRCX", "NEE", "WFC", "TXN", "DHR", "PM", "MCD", "AVGO",
    "IBM", "COST", "HON", "LIN", "UNP", "LLY", "UPS", "SBUX", "MMM", "BLK",
    "QCOM", "GS", "CAT", "BA", "RTX", "AXP", "LOW", "MS", "CVS", "ANTM",
    "BDX", "CHTR", "CI", "TGT", "NOW", "GILD", "DUK", "PLD", "USB", "SPGI",
    "SYK", "D", "ISRG", "SO", "CB", "TMO", "MO", "FIS", "EQIX", "BSX",
    "MMC", "DOW", "VRTX", "ZTS", "BDX", "ADP", "APD", "WBA", "VRTX", "SPG",
    "ISRG", "EW", "MET", "REGN", "TJX", "AON", "CTSH", "PRU", "DE", "ATVI",
    "EOG", "GD", "AEP", "LMT", "FISV", "GM", "CSX", "SCHW", "ECL", "LHX",
    "KMB", "CCI", "WMB", "NSC", "RTX", "KLAC", "NEM", "A", "SO", "ZTS",
    "BKNG", "MMC", "ICE", "ANTM", "IDXX", "STZ", "ROST", "PSX", "TEL", "ROP",
    "AIG", "ESS", "ROST", "PSX", "TEL", "ROP", "AIG", "ESS", "ROST", "PSX",
    "TEL", "ROP", "AIG", "ESS", "REG", "ALL", "DLR", "CBRE", "WLTW", "ESS",
    "CPRT", "WELL", "CFG", "PAYX", "VFC", "AIZ", "AWK", "BKR", "EXC", "AFL",
    "EBAY", "DAL", "ALGN", "VAR", "ED", "CME", "CDNS", "SRE", "WY", "WLTW",
    "ORLY", "TFC", "RF", "ITW", "MTD", "HLT", "VRSK", "WEC", "WST", "HUM",
    "FRC", "ANSS", "BAX", "KEYS", "LH", "AMP", "ALB", "MNST", "ALXN", "MSCI",
    "KMI", "HCA", "HSY", "LUMN", "XEL", "ETR", "EOG", "TRV", "LNT", "PH",
    "LYB", "DLTR", "TDG", "SYY", "AME", "MKC", "STT", "WAB", "CNC", "WDC",
    "MLM", "NVR", "CMI", "BXP", "FTV", "FE", "VRTX", "ANET", "ZBH", "FITB",
    "EXR", "DLR", "UDR", "APH", "ULTA", "RMD", "JKHY", "O", "RCL", "REG",
    "JCI", "IPGP", "QRVO", "TROW", "NUE", "CMS", "ATO", "PGR", "IQV", "RSG",
    "AWK", "MPC", "ESS", "CNC", "AIV", "MTB", "ARE", "HSIC", "ESS", "BKR",
    "GPC", "SYF", "PKG", "DTE", "VMC", "WRB", "ALK", "DFS", "DRI", "HII",
    "TRMB", "KSU", "AMP", "ROL", "SIVB", "PPG", "AEE", "CTXS", "IPG", "LYV",
    "FLT", "PNW", "DVA", "OKE", "CFG", "CTLT", "CINF", "TXT", "IEX", "ZBRA",
    "FLT", "LNT", "LDOS", "EXPD", "KHC", "HSIC", "BIO", "NDAQ", "KMX", "CNP",
    "PPL", "ODFL", "VNO", "LW", "WRK", "GRMN", "BR", "KMX", "APTV", "RE",
    "ALLE", "IP", "BF.B", "AJG", "SJM", "VRSN", "CHD", "ABC", "NRG", "FFIV",
    "K", "IPG", "PNR", "ED", "AVY", "AME", "DVA", "FMC", "COG", "RHI",
    "ED", "SJM", "TSN", "MAS", "HII", "GWW", "AOS", "ZION", "SEE", "OMC",
    "HIG", "LKQ", "ETN", "DISCK", "DOV", "NTAP", "CMA", "AOS", "CAG", "LNC",
    "CTLT", "ESS", "WAB", "MGM", "BIO", "INCY", "CDW", "RJF", "PKG", "HBAN",
    "MAA", "WHR", "FLIR", "TRV", "PFG", "AES", "ETFC", "FRT", "DRE", "NLOK",
    "PFG", "NCLH", "BWA", "WHR", "QRVO", "LW", "TSCO", "VAR", "F", "IRM",
    "PBCT", "CDW", "LDOS", "NLSN", "HSY", "NWL", "BF.B"]

import yfinance as yf
# Fetch historical data for the specified tickers
sp500_data = yf.download(sp500_stocks, start='2015-01-01', end='2023-01-01')

# Display the fetched data
st.date_input("selecet your date of birth")

