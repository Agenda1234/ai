import os
import sys
import asyncio
import requests
from typing import Optional, Dict, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ========== 关键修复：添加项目根目录到Python路径 ==========
# 获取当前文件（weather.py）的目录
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（llm/）
project_root = os.path.dirname(current_file_dir)
# 将根目录添加到sys.path
sys.path.append(project_root)

# 现在可以直接导入mcp_client
from mcp_client import MCPClient

class GlobalWeatherMCPClient(MCPClient):
    WEATHER_CODES: Dict[int, str] = {
        0: "晴朗", 1: "多云", 2: "少云", 3: "阴",
        45: "雾", 48: "霜",
        51: "小雨", 53: "中雨", 55: "大雨",
        61: "小雨", 63: "中雨", 65: "大雨",
        71: "小雪", 73: "中雪", 75: "大雪",
        80: "雷阵雨", 81: "强雷阵雨", 82: "暴雨",
        95: "雷雨", 96: "雷雨加冰雹", 99: "冰雹"
    }

    def __init__(self):
        super().__init__(
            name="weather",
            command="",  # 空命令，避免启动无效进程
            args=[]       # 空参数
        )
        # 本地配置
        self.geocode_timeout = 10
        self.weather_timeout = 10
        # 城市经纬度兜底（解决厦门/深圳等城市编码问题）
        self.city_coords = {
            "深圳": (22.5431, 114.0589),
            "厦门": (24.4700, 118.0800),
            "北京": (39.9042, 116.4074),
            "上海": (31.2304, 121.4737)
        }
        # 中文城市英文映射
        self.city_en_mapping = {
            "深圳": "Shenzhen",
            "厦门": "Xiamen",
            "北京": "Beijing",
            "上海": "Shanghai"
        }

    async def init(self):
        # 手动添加工具列表（替代从服务器加载）
        self.tools = [
            {
                "name": "get_weather",
                "description": "获取指定城市的实时天气信息（支持中英文）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，如：深圳、New York"
                        }
                    },
                    "required": ["city"]
                }
            }
        ]
        print(f"✅ 天气工具初始化完成，加载工具：{[t['name'] for t in self.tools]}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError))
    )
    def geocode_city(self, city_name: str) -> Optional[Tuple[float, float]]:
        # 1. 优先使用硬编码经纬度
        if city_name in self.city_coords:
            return self.city_coords[city_name]
        
        # 2. 尝试英文/拼音搜索
        search_name = self.city_en_mapping.get(city_name, city_name)
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {
            "name": search_name,
            "count": 1,
            "language": "zh",
            "format": "json"
        }

        try:
            response = requests.get(url, params=params, timeout=self.geocode_timeout)
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                # 英文失败，重试中文
                params["name"] = city_name
                response = requests.get(url, params=params, timeout=self.geocode_timeout)
                response.raise_for_status()
                data = response.json()
                if not data.get("results"):
                    return None

            result = data["results"][0]
            return (result["latitude"], result["longitude"])
        except requests.exceptions.RequestException as e:
            print(f"⚠️ 地理编码失败（{city_name}）：{str(e)}")
            return None

    def get_weather_global(self, city_name: str) -> str:
        coords = self.geocode_city(city_name)
        if not coords:
            return f"❌ 无法获取「{city_name}」的地理信息，请检查城市名称"

        lat, lon = coords
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "precipitation", "wind_speed_10m", "weather_code"],
            "timezone": "auto",
            "language": "zh"
        }

        try:
            response = requests.get(url, params=params, timeout=self.weather_timeout)
            response.raise_for_status()
            data = response.json()
            current = data["current"]

            # 解析天气数据
            temp = current["temperature_2m"]
            precipitation = current["precipitation"]
            wind_speed = current["wind_speed_10m"]
            weather_desc = self.WEATHER_CODES.get(current["weather_code"], "未知天气")
            timezone = data["timezone"].split("/")[-1]
            update_time = current["time"].replace("T", " ")

            # 格式化结果
            result = f"""
📌 {city_name} 实时天气
├─ 温度：{temp}°C
├─ 天气：{weather_desc}
├─ 降水量：{precipitation}mm（{"有降水" if precipitation > 0 else "无降水"}）
├─ 风速：{wind_speed}km/h
├─ 时区：{timezone}
└─ 更新时间：{update_time}
            """.strip()
            return result
        except requests.exceptions.RequestException as e:
            return f"❌ 天气查询失败：{str(e)}"
        except Exception as e:
            return f"❌ 数据解析失败：{str(e)}"

    async def call_tool(self, name: str, params: dict):
        """重写工具调用：仅处理本地天气工具"""
        if name != "get_weather":
            return f"❌ 不支持的工具：{name}，仅支持 get_weather"

        city = params.get("city")
        if not city:
            return "❌ 参数错误：缺少必填参数「city」（城市名称）"

        return self.get_weather_global(city)

    def get_tools(self):
        return self.tools

# 测试代码（验证本地模式可用）
if __name__ == "__main__":
    async def test():
        client = GlobalWeatherMCPClient()
        await client.init()
        
        # 测试工具调用
        result = await client.call_tool("get_weather", {"city": "深圳"})
        print(result)

    asyncio.run(test())