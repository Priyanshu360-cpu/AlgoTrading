import requests
# ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1UkM3V0YiLCJqdGkiOiI2YTEwODEwMzkyOGVhYzEwNmRkMjU5ZmIiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaXNFeHRlbmRlZCI6dHJ1ZSwiaWF0IjoxNzc5NDY2NDk5LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE4MTEwMjMyMDB9.NHOGSX7wE0ymFISnEVHU9PAFZRMZSLqoez4qDtix5tY"
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1UkM3V0YiLCJqdGkiOiI2YTEwNzVlMjJhODk5MDVlNzFjYTQxNDMiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzc5NDYzNjUwLCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3ODE5OTI4MDB9.w4JGheLEbV0d1fuqodeeC9qNF3tzMNDTwPhAplcOfVA"
#1)
# url = 'https://api.upstox.com/v2/market-quote/quotes?instrument_key=NSE_EQ%7CINE848E01016'
# headers = {
#     'Content-Type': 'application/json',
#     'Accept': 'application/json',
#     'Authorization': f'Bearer {ACCESS_TOKEN}'
# }

# response = requests.get(url, headers=headers)

# print(response.text)
#Upper one for multiple instruments
#Output - {"status":"success","data":{"NSE_EQ:NHPC":{"ohlc":{"open":79.07,"high":79.3,"low":78.2,"close":78.36},"depth":{"buy":[{"quantity":52880,"price":78.36,"orders":16},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0}],"sell":[{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0}]},"timestamp":"2026-05-22T23:21:57.384+05:30","instrument_token":"NSE_EQ|INE848E01016","symbol":"NHPC","last_price":78.36,"volume":10284991,"average_price":78.64,"oi":0.0,"net_change":-0.64,"total_buy_quantity":52880.0,"total_sell_quantity":0.0,"lower_circuit_limit":71.1,"upper_circuit_limit":86.9,"last_trade_time":"1779445358523","oi_day_high":0.0,"oi_day_low":0.0}}}

#2)
# url = 'https://api.upstox.com/v2/charges/brokerage'
# headers = {
#     'Content-Type': 'application/json',
#     'Accept': 'application/json',
#     'Authorization': f'Bearer {ACCESS_TOKEN}'
# }

# params = {
#     'instrument_token': 'NSE_EQ|INE982J01020',
#     'quantity': '10',
#     'product': 'I',
#     'transaction_type': 'BUY',
#     'price': '1000'
# }

# response = requests.get(url, headers=headers, params=params)

# print(response.json())
# THIS API IS FOR BROKERAGE CALCULATION
#Output - {'status': 'success', 'data': {'charges': {'total': 7.76, 'brokerage': 6.0, 'taxes': {'gst': 1.14, 'stt': 0.0, 'stamp_duty': 0.3}, 'otherTaxes': {'transaction': 0.31, 'clearing': 0.0, 'ipft': 0.0, 'sebi_turnover': 0.01}, 'dpPlan': {'name': 'DP4A', 'min_expense': 20.0}, 'other_charges': {'transaction': 0.31, 'clearing': 0.0, 'ipft': 0.0, 'sebi_turnover': 0.01}, 'dp_plan': {'name': 'DP4A', 'min_expense': 20.0}}}}

#3)
# url = 'https://api.upstox.com/v2/market/status/NSE'
# headers = {
#     'Content-Type': 'application/json',
#     'Accept': 'application/json',
#     'Authorization': f'Bearer {ACCESS_TOKEN}'
# }

# response = requests.get(url, headers=headers)

# print(response.json())

#This API is for checking market status
#Output - {'status': 'success', 'data': {'exchange': 'NSE', 'status': 'CLOSING_END', 'last_updated': 1779445800000}}


# url = 'https://api.upstox.com/v2/market-quote/quotes?instrument_key=NSE_EQ%7CINE848E01016'
# headers = {
#     'Content-Type': 'application/json',
#     'Accept': 'application/json',
#     'Authorization': f'Bearer {ACCESS_TOKEN}'
# }

# response = requests.get(url, headers=headers)

# print(response.text)
#This Api is for fetching market quote for a particular instrument
#Output - {"status":"success","data":{"NSE_EQ:NHPC":{"ohlc":{"open":79.07,"high":79.3,"low":78.2,"close":78.36},"depth":{"buy":[{"quantity":52880,"price":78.36,"orders":16},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0}],"sell":[{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0},{"quantity":0,"price":0.0,"orders":0}]},"timestamp":"2026-05-23T08:00:41.497+05:30","instrument_token":"NSE_EQ|INE848E01016","symbol":"NHPC","last_price":78.36,"volume":10284991,"average_price":78.64,"oi":0.0,"net_change":-0.64,"total_buy_quantity":52880.0,"total_sell_quantity":0.0,"lower_circuit_limit":70.53,"upper_circuit_limit":86.19,"last_trade_time":"1779445358523","oi_day_high":0.0,"oi_day_low":0.0}}}

#4)

# import requests

# url = 'https://api-sandbox.upstox.com/v3/order/place'
# headers = {
#     'Content-Type': 'application/json',
#     'Accept': 'application/json',
#     'Authorization': f'Bearer {ACCESS_TOKEN}',
# }

# data = {
#     'quantity': 25,
#     'product': 'D',
#     'validity': 'DAY',
#     'price': 0,
#     'tag': 'string',
#     'instrument_token': 'NSE_EQ|INE982J01020',
#     'order_type': 'MARKET',
#     'transaction_type': 'BUY',
#     'disclosed_quantity': 0,
#     'trigger_price': 0,
#     'is_amo': False,
#     'slice': False
# }

# try:
#     # Send the POST request
#     response = requests.post(url, json=data, headers=headers)

#     # Print the response status code and body
#     print('Response Code:', response.status_code)
#     print('Response Body:', response.json())

# except Exception as e:
#     # Handle exceptions
#     print('Error:', str(e))

#This API is for placing an order
#Output -Response Body: {'status': 'success', 'data': {'order_ids': ['260523092600030']}, 'metadata': {'latency': 3}}

#5) 

# url = 'https://api-sandbox.upstox.com/v2/order/details'
# headers = {
#     'Content-Type': 'application/json',
#     'Accept': 'application/json',
#     'Authorization': f'Bearer {ACCESS_TOKEN}'
# }

# params = {'order_id': '260523092600030'}

# response = requests.get(url, headers=headers, params=params)

# print(response.text)

#This API is for fetching order details
#Output - {"status":"success","data":{"exchange":"NSE","product":"D","price":0.0,"quantity":25,"status":"open pending","guid":null,"tag":null,"instrument_token":"NSE_EQ|INE982J01020","placed_by":"5RC7WF","tradingsymbol":"PAYTM","trading_symbol":"PAYTM","order_type":"MARKET","validity":"DAY","trigger_price":0.0,"disclosed_quantity":0,"transaction_type":"BUY","average_price":0.0,"filled_quantity":0,"pending_quantity":0,"status_message":null,"status_message_raw":null,"exchange_order_id":"","parent_order_id":null,"order_id":"260523092600030","variety":"","order_timestamp":"2026-05-23 09:26:00","exchange_timestamp":null,"is_amo":false,"order_request_id":"UDAPI-99fcf11d-67e1-44e2-863d-14474fff1a45","order_ref_id":""}}

#6)

# url = 'https://api-sandbox.upstox.com/v2/order/retrieve-all'
# headers = {
#     'Content-Type': 'application/json',
#     'Accept': 'application/json',
#     'Authorization': f'Bearer {ACCESS_TOKEN}'
# }

# response = requests.get(url, headers=headers)
# print(response.text)

#This API is for fetching all orders
#Output - {"status":"success","data":[{"exchange":"NSE","product":"D","price":0.0,"quantity":25,"status":"open pending","guid":null,"tag":null,"instrument_token":"NSE_EQ|INE982J01020","placed_by":"5RC7WF","tradingsymbol":"PAYTM","trading_symbol":"PAYTM","order_type":"MARKET","validity":"DAY","trigger_price":0.0,"disclosed_quantity":0,"transaction_type":"BUY","average_price":0.0,"filled_quantity":0,"pending_quantity":0,"status_message":null,"status_message_raw":null,"exchange_order_id":"","parent_order_id":null,"order_id":"260523092553927","variety":"","order_timestamp":"2026-05-23 09:25:53","exchange_timestamp":null,"is_amo":false,"order_request_id":"UDAPI-4eb89353-ab1e-4a0a-b582-a2b9c0c29820","order_ref_id":""},{"exchange":"NSE","product":"D","price":0.0,"quantity":25,"status":"open pending","guid":null,"tag":null,"instrument_token":"NSE_EQ|INE982J01020","placed_by":"5RC7WF","tradingsymbol":"PAYTM","trading_symbol":"PAYTM","order_type":"MARKET","validity":"DAY","trigger_price":0.0,"disclosed_quantity":0,"transaction_type":"BUY","average_price":0.0,"filled_quantity":0,"pending_quantity":0,"status_message":null,"status_message_raw":null,"exchange_order_id":"","parent_order_id":null,"order_id":"260523092600030","variety":"","order_timestamp":"2026-05-23 09:26:00","exchange_timestamp":null,"is_amo":false,"order_request_id":"UDAPI-99fcf11d-67e1-44e2-863d-14474fff1a45","order_ref_id":""}]}

#7)
# url = 'https://api-sandbox.upstox.com/v3/order/modify'
# headers = {
#     'Accept': 'application/json',
#     'Content-Type': 'application/json',
#     'Authorization': f'Bearer {ACCESS_TOKEN}'
# }

# data = {
#     'quantity': 3,
#     'validity': 'DAY',
#     'price': 1050,
#     'order_id': '260523092553927',
#     'order_type': 'LIMIT',
#     'disclosed_quantity': 0,
#     'trigger_price': 1050,
# }

# response = requests.put(url, headers=headers, json=data)
# print(response.text)

#This API is for modifying an order
#Output - {"status":"success","data":{"order_id":"260523092553927"},"metadata":{"latency":20}}

#7)
# url = 'https://api-sandbox.upstox.com/v3/order/cancel?order_id=260523092553927'
# headers = {
#     'Content-Type': 'application/json',
#     'Accept': 'application/json',
#     'Authorization': f'Bearer {ACCESS_TOKEN}'
# }

# response = requests.delete(url, headers=headers)

# print(response.text)
#This API is for cancelling an order
#Output -{"status":"success","data":{"order_id":"260523092553927"},"metadata":{"latency":5}}

#8 this API is for exiting all positions but it is not working in sandbox environment
# url = 'https://api-sandbox.upstox.com/v2/order/positions/exit'
# headers = {
#     'Content-Type': 'application/json',
#     'Accept': 'application/json',
#     'Authorization': f'Bearer {ACCESS_TOKEN}',
# }

# data = {}

# try:
#     # Send the POST request
#     response = requests.post(url, json=data, headers=headers)

#     # Print the response status code and body
#     print('Response Code:', response.status_code)
#     print('Response Body:', response.json())

# except Exception as e:
#     # Handle exceptions
#     print('Error:', str(e))

#9)
# url = 'https://api-sandbox.upstox.com/v2/news'
# params = {
#     'category': 'instrument_keys',
#     'instrument_keys': 'NSE_EQ|INE040H01021,NSE_EQ|INE002A01018'
# }
# headers = {
#     'Accept': 'application/json',
#     'Authorization': f'Bearer {ACCESS_TOKEN}'
# }

# response = requests.get(url, params=params, headers=headers)

# print(response.json())
#This API is for fetching news related to particular instruments


url = 'https://api-sandbox.upstox.com/v2/portfolio/short-term-positions'
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Bearer {ACCESS_TOKEN}'
}

response = requests.get(url, headers=headers)

print(response.json())
