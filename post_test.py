import requests
import json

# 定义接口的基础 URL
base_url = 'http://127.0.0.1:5000'

# 设置请求头
headers = {
    'Content-Type': 'application/json'
}

# 定义一个函数来流式调用 /chat 接口
def call_chat_api(user_id, question):
    url = f'{base_url}/chat'
    data = {
        "user_id": user_id,
        "question": question
    }
    json_data = json.dumps(data)
    try:
        response = requests.post(url, headers=headers, data=json_data, stream=True)
        if response.status_code == 200:
            print("请求成功，响应内容如下：")
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        chunk = json.loads(line)
                        if 'chunk' in chunk:
                            print(chunk['chunk'], end='', flush=True)
                        elif 'error' in chunk:
                            print(f"发生错误：{chunk['error']}")
                    except json.JSONDecodeError:
                        print("解析响应数据时出错")
            print()
        else:
            print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")
    except requests.RequestException as e:
        print(f"请求发生错误：{e}")

# 定义一个函数来调用 /clear_history 接口
def call_clear_history_api(user_id):
    url = f'{base_url}/clear_history'
    data = {
        "user_id": user_id
    }
    json_data = json.dumps(data)
    try:
        response = requests.post(url, headers=headers, data=json_data)
        if response.status_code == 200:
            result = response.json()
            print("请求成功，响应内容如下：")
            print(result)
        else:
            print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")
    except requests.RequestException as e:
        print(f"请求发生错误：{e}")

# 主循环
while True:
    print("\n请选择操作：")
    print("1. 发起聊天请求")
    print("2. 清空聊天历史记录")
    print("3. 退出程序")
    choice = input("请输入操作编号：")

    if choice == '1':
        user_id = input("请输入用户 ID：")
        while True:
            question = input("请输入问题（输入 'exit' 退出当前聊天操作）：")
            if question.lower() == 'exit':
                break
            call_chat_api(user_id, question)
    elif choice == '2':
        user_id = input("请输入要清空历史记录的用户 ID：")
        call_clear_history_api(user_id)
    elif choice == '3':
        print("程序已退出。")
        break
    else:
        print("无效的选择，请重新输入。")
