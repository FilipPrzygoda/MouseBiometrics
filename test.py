import requests

cookies = {'user_id': 'filipp'}
events = [
    {"type": "mousemove", "score": 1, "x": 100, "y": 100, "timestamp": 10},
    {"type": "mousemove", "score": 1, "x": 110, "y": 110, "timestamp": 20},
    {"type": "diamond_click", "score": 1, "x": 110, "y": 110, "timestamp": 30, "diamondLeft": 100, "diamondTop": 100}

]
r = requests.post("http://localhost:5000/api/recognize", json={"events": events}, cookies=cookies)
print(r.status_code, r.text)

