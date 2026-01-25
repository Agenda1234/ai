import sys
import os
import asyncio
import warnings
import traceback
import uuid
from datetime import datetime

# ========== 核心修复1：正确添加项目路径 ==========
# 获取当前文件（app.py）的绝对路径
current_file = os.path.abspath(__file__)
# 获取当前文件所在目录（html/）
current_dir = os.path.dirname(current_file)
# 获取项目根目录（demo/）
demo_dir = os.path.dirname(current_dir)
# 获取顶级目录（ai/1/）- 确保能找到llm目录
root_dir = os.path.dirname(demo_dir)
# 添加根目录到系统路径（关键：让Python能找到llm包）
sys.path.append(root_dir)
sys.path.append(demo_dir)

# ========== 导入Flask及相关模块 ==========
from flask import Flask, render_template, request, make_response, redirect, url_for, jsonify

# ========== 核心修复2：健壮的LLM导入+异步封装 ==========
llm_main = None

# 最终兜底：手动添加llm目录路径
llm_dir = os.path.join(demo_dir, "llm")
sys.path.append(llm_dir)
try:
    from main import main as llm_main
except Exception as e:
    raise ImportError(f"❌ 无法导入llm.main模块：{str(e)}")

def run_llm(user_input):
    """
    统一的LLM调用入口：兼容同步/异步main函数
    :param user_input: 用户输入文本
    :return: 机器人回复字符串
    """
    if not user_input or not user_input.strip():
        return "⚠️ 输入不能为空！"
    
    try:
        # 判断是否为异步函数
        if asyncio.iscoroutinefunction(llm_main):
            # 修复：创建新的事件循环（解决Flask debug模式下的循环冲突）
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 运行异步函数
            result = loop.run_until_complete(llm_main(user_input))
        else:
            # 同步函数直接调用
            result = llm_main(user_input)
        
        # 结果格式化
        if result is None:
            return "🤖 抱歉，我暂时没有找到相关答案。"
        return str(result).strip()
    except Exception as e:
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ LLM调用失败：{error_detail}")
        return f"<span style='color:#f56c6c;'>🤖 调用失败：{str(e)}</span>"

# ========== Flask应用初始化 ==========
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # 更安全的随机密钥
app.config['JSON_AS_ASCII'] = False  # 支持中文JSON输出
app.config['TEMPLATES_AUTO_RELOAD'] = True  # 模板自动重载

# ========== 全局变量：聊天记录（优化存储结构） ==========
chat_records = []  # 格式：[{
#     'id': '唯一ID',
#     'role': 'user/bot',
#     'content': '消息内容',
#     'time': 'HH:MM:SS',
#     'timestamp': 时间戳（用于排序）
# }]

# ========== 工具函数 ==========
def format_message_content(content):
    """格式化消息内容：支持换行、空格、基础Markdown"""
    if not content:
        return ""
    # 转换换行和空格
    content = str(content).replace('\n', '<br>').replace(' ', '&nbsp;')
    # 简单的Markdown支持（加粗、链接）
    content = content.replace('**', '<strong>').replace('__', '</strong>')
    content = content.replace('[', '<a href="').replace('](', '">').replace(')', '</a>')
    return content

# ========== 路由定义 ==========
@app.route('/', methods=['GET', 'POST'])
def chat():
    global chat_records
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if request.method == 'POST':
        # 修复3：防重复提交（通过请求ID+内容双重校验）
        msg_id = request.form.get('msg_id', str(uuid.uuid4()))
        user_input = request.form.get('message', '').strip()
        
        # 校验：非空 + 未提交过
        if user_input and not any(
            msg['id'] == msg_id or 
            (msg['role'] == 'user' and msg['content'] == user_input and 
             abs(msg['timestamp'] - datetime.now().timestamp()) < 3)  # 3秒内相同内容去重
            for msg in chat_records
        ):
            # 1. 添加用户消息到聊天记录
            user_msg = {
                'id': msg_id,
                'role': 'user',
                'content': format_message_content(user_input),
                'time': current_time,
                'timestamp': datetime.now().timestamp()
            }
            chat_records.append(user_msg)
            
            # 2. 调用LLM并获取回复
            bot_reply = run_llm(user_input)
            bot_reply_formatted = format_message_content(bot_reply)
            
            # 3. 添加机器人回复
            bot_msg = {
                'id': str(uuid.uuid4()),
                'role': 'bot',
                'content': bot_reply_formatted,
                'time': current_time,
                'timestamp': datetime.now().timestamp()
            }
            chat_records.append(bot_msg)
    
    # 渲染页面：添加缓存控制，避免历史记录加载异常
    resp = make_response(render_template('chat.html', chats=chat_records))
    resp.headers.update({
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    })
    return resp

# ========== AJAX接口：异步发送消息（无刷新） ==========
@app.route('/send_msg', methods=['POST'])
def send_msg():
    try:
        # 获取JSON数据（兼容form-data）
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        user_input = data.get('message', '').strip()
        msg_id = data.get('msg_id', str(uuid.uuid4()))
        
        # 输入校验
        if not user_input:
            return jsonify({
                'code': 400,
                'msg': '输入不能为空',
                'data': None
            })
        
        # 防重复提交
        if any(msg['id'] == msg_id for msg in chat_records):
            return jsonify({
                'code': 409,
                'msg': '消息已提交，请勿重复发送',
                'data': None
            })
        
        # 调用LLM
        current_time = datetime.now().strftime("%H:%M:%S")
        bot_reply = run_llm(user_input)
        bot_reply_formatted = format_message_content(bot_reply)
        
        # 构造返回数据
        return jsonify({
            'code': 200,
            'msg': 'success',
            'data': {
                'msg_id': msg_id,
                'user_msg': {
                    'content': format_message_content(user_input),
                    'time': current_time
                },
                'bot_msg': {
                    'content': bot_reply_formatted,
                    'time': current_time
                }
            }
        })
    except Exception as e:
        error_msg = f"调用失败：{str(e)}"
        print(f"【AJAX错误详情】:\n{traceback.format_exc()}")
        return jsonify({
            'code': 500,
            'msg': error_msg,
            'data': None
        })

# ========== 清空聊天记录 ==========
@app.route('/clear', methods=['POST'])
def clear_chat():
    global chat_records
    chat_records = []
    return redirect(url_for('chat'))

# ========== AJAX清空聊天记录 ==========
@app.route('/clear_ajax', methods=['POST'])
def clear_ajax():
    global chat_records
    chat_records = []
    return jsonify({
        'code': 200,
        'msg': '聊天记录已清空',
        'data': None
    })

# ========== 健康检查接口 ==========
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'code': 200,
        'status': 'running',
        'timestamp': datetime.now().timestamp(),
        'chat_count': len(chat_records)
    })

# ========== 主函数 ==========
if __name__ == '__main__':
    print(f"✅ 项目根目录：{root_dir}")
    print(f"✅ Demo目录：{demo_dir}")
    print(f"✅ LLM模块导入状态：{'成功' if llm_main else '失败'}")
    print("🚀 Flask服务启动中... http://127.0.0.1:5000")
    
    # 修复4：解决debug模式下的异步循环冲突
    if app.config['DEBUG']:
        asyncio.set_event_loop(asyncio.new_event_loop())
    
    # 启动服务（关闭debug时建议用host='0.0.0.0'允许外部访问）
    app.run(
        debug=True, 
        port=5000, 
        host='127.0.0.1',
        use_reloader=False  # 关闭自动重载（避免异步循环问题）
    )