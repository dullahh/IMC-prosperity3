import json
from typing import Any, Dict, List
import math
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


import json
import math
from typing import Dict, List
from datamodel import Order, Symbol, Trade, TradingState


class Trader:
    def __init__(self):
        self.alpha = 0.2
        self.window = 20

        self.volcano_strikes = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500,
        }

        self.position_limits = {
            "KELP": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "RAINFOREST_RESIN": 100,
            "VOLCANIC_ROCK": 400,
            **{k: 200 for k in self.volcano_strikes},
        }

        self.volatility = 0.3
        self.risk_free_rate = 0.01

    def black_scholes_call(self, S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
        return S * N(d1) - K * math.exp(-r * T) * N(d2)

    def rainforest_resin(self, position):
        product = "RAINFOREST_RESIN"
        orders = []
        p = 47
        our_bid_vol = 50 - position
        our_ask_vol = -50 - position
        if abs(position) < p:
            if our_bid_vol > 0:
                orders.append(Order(product, 9998, our_bid_vol))
                logger.print(f"{product}: BUY {our_bid_vol} @ 9998")
            if our_ask_vol < 0:
                orders.append(Order(product, 10002, our_ask_vol))
                logger.print(f"{product}: SELL {-our_ask_vol} @ 10002")
        elif position >= p:
            dv = 3 if position < 48 else 5
            orders.append(Order(product, 9996, 50 - position))
            orders.append(Order(product, 10000, -dv))
            orders.append(Order(product, 10002, -50 - position + dv))
        elif position <= -p:
            dv = 3 if position > -48 else 5
            orders.append(Order(product, 10000, dv))
            orders.append(Order(product, 9998, 50 - position - dv))
            orders.append(Order(product, 10004, -50 - position))
        return orders

    def vwap(self, order_dict, depth=3, is_bid=True):
        sorted_orders = sorted(order_dict.items(), reverse=is_bid)[:depth]
        total_volume = sum(abs(v) for _, v in sorted_orders)
        total_price = sum(p * abs(v) for p, v in sorted_orders)
        return total_price / total_volume if total_volume else None

    def update_ema(self, price, ema):
        return price if ema is None else self.alpha * price + (1 - self.alpha) * ema

    def market_make(self, product, fair, position, limit, edge=2):
        orders = []
        bid = fair - edge
        ask = fair + edge
        if position > limit / 2:
            ask -= 1
        elif position < -limit / 2:
            bid += 1
        buy_vol = max(0, limit - position)
        sell_vol = max(0, position + limit)
        if buy_vol > 0:
            orders.append(Order(product, round(bid), min(5, buy_vol)))
            logger.print(f"{product}: BUY {min(5, buy_vol)} @ {round(bid)}")
        if sell_vol > 0:
            orders.append(Order(product, round(ask), -min(5, sell_vol)))
            logger.print(f"{product}: SELL {min(5, sell_vol)} @ {round(ask)}")
        return orders

    def handle_kelp(self, product, order_depth, market_trades, position, position_limit, history, result):
        orders = []
        window = 10
        if len(history) < window:
            return
        kelp_mean = sum(history[-window:]) / window
        kelp_std = (sum((p - kelp_mean) ** 2 for p in history[-window:]) / window) ** 0.5
        buy_thresh = kelp_mean - 0.05 * kelp_std
        sell_thresh = kelp_mean + 0.05 * kelp_std

        buy_orders = order_depth.buy_orders
        sell_orders = order_depth.sell_orders
        best_bid = max(buy_orders) if buy_orders else None
        best_ask = min(sell_orders) if sell_orders else None

        if best_bid is None or best_ask is None:
            return

        trade_volume = min(abs(buy_orders.get(best_bid, 5)), abs(sell_orders.get(best_ask, 5)))
        dynamic_volume = min((trade_volume + (best_ask - best_bid)), position_limit)

        if best_ask <= buy_thresh and position < position_limit:
            volume = min(-sell_orders.get(best_ask, 0), position_limit - position, int(dynamic_volume))
            if volume > 0:
                orders.append(Order(product, best_ask, volume))
                logger.print(f"KELP: BUY {volume} @ {best_ask}")

        if best_bid >= sell_thresh and position > -position_limit:
            volume = min(buy_orders.get(best_bid, 0), position + position_limit, int(dynamic_volume))
            if volume > 0:
                orders.append(Order(product, best_bid, -volume))
                logger.print(f"KELP: SELL {volume} @ {best_bid}")

        result[product] = orders

    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = state.traderData or "{}"
        try:
            price_history = json.loads(trader_data)
        except:
            price_history = {}

        rock_price = None
        rock_trades = state.market_trades.get("VOLCANIC_ROCK", [])
        if rock_trades:
            rock_price = sum(t.price for t in rock_trades) / len(rock_trades)

        for product, order_depth in state.order_depths.items():
            if not order_depth.buy_orders or not order_depth.sell_orders:
                continue

            position = state.position.get(product, 0)
            buy_orders = order_depth.buy_orders
            sell_orders = order_depth.sell_orders

            if product in self.volcano_strikes and rock_price is not None:
                strike = self.volcano_strikes[product]
                best_ask = min(sell_orders)
                best_bid = max(buy_orders)
                limit = self.position_limits[product]
                T = max(1, 7 - state.timestamp // 1000) / 7.0
                fair = self.black_scholes_call(rock_price, strike, T, self.risk_free_rate, self.volatility)
                orders = []

                # Swapped logic (buy when overpriced, sell when underpriced)
                if best_ask > fair and position < limit:
                    volume = min(-sell_orders[best_ask], limit - position)
                    if volume > 0:
                        orders.append(Order(product, best_ask, volume))
                        logger.print(f"{product}: BUY {volume} @ {best_ask} > fair {fair:.2f}")
                if best_bid < fair and position > -limit:
                    volume = min(buy_orders[best_bid], position + limit)
                    if volume > 0:
                        orders.append(Order(product, best_bid, -volume))
                        logger.print(f"{product}: SELL {volume} @ {best_bid} < fair {fair:.2f}")

                result[product] = orders
                continue

            if product == "RAINFOREST_RESIN":
                result[product] = self.rainforest_resin(position)
                continue

            if product == "KELP":
                market_trades = state.market_trades.get(product, [])
                trade_prices = [t.price for t in market_trades]
                history_key = f"{product}_HIST"
                history = price_history.get(history_key, [])
                history.extend(trade_prices)
                history = history[-self.window:]
                price_history[history_key] = history
                self.handle_kelp(product, order_depth, market_trades, position, self.position_limits[product], history, result)
                continue

            if product not in self.position_limits:
                continue

            market_trades = state.market_trades.get(product, [])
            trade_prices = [t.price for t in market_trades]
            history_key = f"{product}_HIST"
            ema_key = f"{product}_EMA"
            history = price_history.get(history_key, [])
            ema = price_history.get(ema_key)
            history.extend(trade_prices)
            history = history[-self.window:]
            price_history[history_key] = history
            vwap_bid = self.vwap(buy_orders)
            vwap_ask = self.vwap(sell_orders, is_bid=False)
            if vwap_bid is None or vwap_ask is None:
                continue
            mid = (vwap_bid + vwap_ask) / 2
            ema = self.update_ema(mid, ema)
            price_history[ema_key] = ema
            result[product] = self.market_make(product, ema, position, self.position_limits[product])

        trader_data_out = json.dumps(price_history)
        logger.flush(state, result, conversions, trader_data_out)
        return result, conversions, trader_data_out
