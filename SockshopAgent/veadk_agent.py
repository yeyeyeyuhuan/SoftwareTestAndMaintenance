import os
import time
import json
import requests
import subprocess
from openai import OpenAI

# =========================
# 全局配置
# =========================
PROMETHEUS_URL = "http://127.0.0.1:63547"   # 改成你自己的 Prometheus 地址
THRESHOLD = 0.005                           # 当前触发阈值
MODEL_NAME = "gpt-4o"                       # 改成你实际可用模型
BASE_URL = "https://www.nkd230gzs.xyz/v1"      # 改成你实际可用 API 根地址
NAMESPACE = "sock-shop"
TARGET_SERVICE = "catalogue"

# ==========================================
# 第一部分：定义 Agent 的工具箱
# ==========================================

def execute_promql(query_str):
    """工具 1：执行 PromQL 查询获取指标"""
    print(f"查询监控: {query_str}")
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query_str},
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("data", {}).get("result", [])
        return str(results) if results else "未查询到数据"
    except Exception as e:
        return f"查询失败: {e}"

def get_pod_name_by_service(service_name):
    """
    根据 deployment/service 名获取一个 pod 名
    这里用 app=<service_name> 作为常见 label，
    如果不适配你的环境，我们再改。
    """
    try:
        cmd = [
            "kubectl", "get", "pods",
            "-n", NAMESPACE,
            "-l", f"name={service_name}",
            "-o", "jsonpath={.items[0].metadata.name}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        pod_name = result.stdout.strip()
        if pod_name:
            return pod_name

        # 兜底：按 pod 名模糊匹配
        cmd2 = ["kubectl", "get", "pods", "-n", NAMESPACE, "--no-headers"]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        lines = result2.stdout.strip().splitlines()
        for line in lines:
            parts = line.split()
            if parts and parts[0].startswith(service_name):
                return parts[0]

        return None
    except Exception:
        return None

def get_service_logs(service_name, tail_lines=20):
    """工具 2：抓取真实 Pod 日志"""
    print(f"抓取日志: {service_name} (最后 {tail_lines} 行)")
    try:
        pod_name = get_pod_name_by_service(service_name)
        if not pod_name:
            return f"未找到服务 {service_name} 对应的 Pod"

        cmd = [
            "kubectl", "logs", pod_name,
            "-n", NAMESPACE,
            f"--tail={tail_lines}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout.strip()
            return output if output else "日志为空"
        return f"日志抓取失败: {result.stderr.strip()}"
    except Exception as e:
        return f"抓取日志失败: {e}"

def restart_pod(service_name):
    """工具 3：自动恢复，重启 deployment"""
    print(f"正在重启服务: {service_name} ...")
    try:
        cmd = [
            "kubectl", "rollout", "restart",
            f"deployment/{service_name}",
            "-n", NAMESPACE
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return f"{service_name} 服务已成功触发重启。"
        return f"重启失败: {result.stderr.strip()}"
    except Exception as e:
        return f"重启失败: {e}"

AVAILABLE_TOOLS = {
    "execute_promql": execute_promql,
    "get_service_logs": get_service_logs,
    "restart_pod": restart_pod
}

# ==========================================
# 第二部分：Agent
# ==========================================

class AIOpsAgent:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL
        )

        self.system_prompt = """
你是一个资深的云原生 AIOps 专家，负责 Sock Shop 微服务系统的异常诊断与恢复。

你的工作要求：
1. 当收到异常告警时，先分析上下文；
2. 必要时调用工具收集证据，包括 Prometheus 指标和服务日志；
3. 基于证据判断异常是否严重；
4. 如果确认服务处于明显异常状态，并且重启有助于快速恢复，请直接调用 restart_pod 工具执行恢复；
5. 最后输出简洁的诊断报告，说明异常现象、分析依据、采取的动作和恢复建议。

请尽量自主完成诊断与恢复，不要只停留在“建议重启”，而是直接执行工具。
"""

        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "execute_promql",
                    "description": "执行 PromQL 获取 Prometheus 监控指标",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_str": {"type": "string"}
                        },
                        "required": ["query_str"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_service_logs",
                    "description": "获取指定微服务最新日志",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_name": {"type": "string"},
                            "tail_lines": {"type": "integer"}
                        },
                        "required": ["service_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "restart_pod",
                    "description": "当服务出现严重异常且需要快速恢复时，重启目标服务",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_name": {"type": "string"}
                        },
                        "required": ["service_name"]
                    }
                }
            }
        ]

    def run_diagnosis(self, alert_context):
        print("\n" + "=" * 50)
        print(f"[Agent 唤醒] 接收到异常上下文: {alert_context}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"系统出现异常：{alert_context}。请进行排查并在必要时执行恢复。"}
        ]

        for step in range(6):
            print(f"[Agent 思考中 - 第 {step + 1} 步]...")
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=self.tools_schema,
                tool_choice="auto"
            )

            response_message = response.choices[0].message

            # 把 assistant 回复写回上下文
            assistant_msg = {
                "role": "assistant",
                "content": response_message.content or ""
            }

            if response_message.tool_calls:
                assistant_msg["tool_calls"] = []
                for tool_call in response_message.tool_calls:
                    assistant_msg["tool_calls"].append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })

            messages.append(assistant_msg)

            # 如果模型要求调用工具
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    function_to_call = AVAILABLE_TOOLS[function_name]
                    tool_result = function_to_call(**function_args)

                    print(f"[工具返回] {function_name}: {tool_result}")

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_result),
                    })
            else:
                print("\n[Agent 最终诊断报告]:")
                print(response_message.content)
                print("=" * 50 + "\n")
                return

        print("\n[Agent] 达到最大推理步数，诊断结束。")
        print("=" * 50 + "\n")


# ==========================================
# 第三部分：主程序
# ==========================================

def fetch_basic_cpu():
    """轻量级巡检：查 catalogue CPU"""
    promql = 'sum(rate(container_cpu_usage_seconds_total{namespace="sock-shop", pod=~"catalogue.*"}[1m]))'
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": promql},
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("data", {}).get("result", [])
        return float(results[0]["value"][1]) if results else 0.0
    except Exception:
        return 0.0

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请先配置 OPENAI_API_KEY 环境变量")

    agent = AIOpsAgent(api_key=api_key)

    print("智能监控守护进程已启动...")
    while True:
        cpu_load = fetch_basic_cpu()
        print(f"[{time.strftime('%H:%M:%S')}] 日常巡检 | Catalogue CPU 负载: {cpu_load:.4f}")

        if cpu_load > THRESHOLD:
            alert_msg = f"Sock Shop 的 {TARGET_SERVICE} 服务 CPU 突增至 {cpu_load:.4f}，超过阈值 {THRESHOLD}"
            agent.run_diagnosis(alert_context=alert_msg)

            # 诊断后休眠，避免重复触发
            time.sleep(60)

        time.sleep(10)

if __name__ == "__main__":
    main()