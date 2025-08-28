# IMC-prosperity3
My first ever go with algorithmic trading in a competitive environment. This scripts contains my code for the algo trading side
The main part of this code is the Trader class, the logger one can be disregarded as it a provided class used to help visualise your trading performance




 This Trader class implements an automated trading strategy for the IMC Prosperity Challenge, combining market-making, mean reversion, and option pricing techniques. It dynamically generates buy/sell orders for different products while respecting position limits and managing risk.

Key Features

  General Structure
  
    Maintains price history with exponential moving averages (EMA).
    
    Dynamically adjusts bids/asks based on inventory levels.
    
    Respects per-product position limits to manage exposure.
    
    Stores state across iterations using serialized traderData.
  
  Product-Specific Strategies
  
    Volcanic Rock Vouchers (options)
    
    Uses a Black-Scholes call pricing model with assumed volatility and risk-free rate.
    
    Compares market prices to theoretical fair value.
    
    Buys when the option is overpriced, sells when underpriced (arbitrage-style strategy).
    
    Rainforest Resin
    
    Implements a custom inventory-driven quoting strategy:
    
    Places tight bid/ask spreads when near neutral position.
    
    Widens spreads and scales volume when inventory approaches limits.
    
    Kelp
    
    Statistical mean reversion strategy:
    
    Tracks a rolling window of prices.
    
    Calculates mean & standard deviation.
    
    Buys below a lower threshold, sells above an upper threshold.
  
  Other Products
  
    Runs a VWAP-based market-making strategy using EMA mid-prices.
    
    Adjusts quotes slightly up or down depending on current inventory.
    
    Utilities
    
    vwap(): Computes volume-weighted average price for bids/asks.
    
    update_ema(): Maintains smoothed EMA signals.
    
    market_make(): Places symmetric bid/ask orders around estimated fair price.
    
    black_scholes_call(): Black-Scholes formula for theoretical option pricing.

Throughout the competition, my main drivers were  KELP(a stable security that oscillates a lot) and VOLCANIC ROCK(a very volatile security)
