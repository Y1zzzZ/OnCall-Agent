import socket
import requests

def check_port(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        try:
            s.connect((host, port))
            return True
        except:
            return False

def check_http(url):
    try:
        # Try POST to /mcp/tools/list
        r = requests.post(url, json={}, timeout=5)
        print(f"POST {url} -> Status: {r.status_code}")
        print(f"Content: {r.text[:500]}")
    except Exception as e:
        print(f"POST {url} -> Error: {e}")

def main(host="127.0.0.1", port=8080):
    if check_port(host, port):
        print(f"Port {port} is OPEN.")
        check_http(f"http://{host}:{port}/mcp/tools/list")
        check_http(f"http://{host}:{port}/tools/list")
    else:
        print(f"Port {port} is CLOSED.")

if __name__ == "__main__":
    main(port=8085)
