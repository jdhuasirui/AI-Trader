#!/usr/bin/env python3
"""
Alpaca MCP Service Startup Script

Starts all MCP services required for Alpaca paper trading:
- Math service (calculations)
- Search service (news)
- Alpaca Price service (real-time quotes)
- Alpaca Trade service (order execution)
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class AlpacaMCPServiceManager:
    """Manager for Alpaca-related MCP services"""

    def __init__(self):
        self.services = {}
        self.running = True

        # Set default ports
        self.ports = {
            "math": int(os.getenv("MATH_HTTP_PORT", "8000")),
            "search": int(os.getenv("SEARCH_HTTP_PORT", "8001")),
            "alpaca_price": int(os.getenv("ALPACA_PRICE_PORT", "8010")),
            "alpaca_trade": int(os.getenv("ALPACA_TRADE_PORT", "8011")),
            "alpaca_news": int(os.getenv("ALPACA_NEWS_PORT", "8012")),
            "trade_flow": int(os.getenv("TRADE_FLOW_PORT", "8013")),
            "corporate_actions": int(os.getenv("CORPORATE_ACTIONS_PORT", "8014")),
            "onchain": int(os.getenv("ONCHAIN_PORT", "8015")),
        }

        # Service configurations
        mcp_server_dir = os.path.dirname(os.path.abspath(__file__))
        self.service_configs = {
            "math": {
                "script": os.path.join(mcp_server_dir, "tool_math.py"),
                "name": "Math",
                "port": self.ports["math"],
            },
            "search": {
                "script": os.path.join(mcp_server_dir, "tool_alphavantage_news.py"),
                "name": "Search",
                "port": self.ports["search"],
            },
            "alpaca_price": {
                "script": os.path.join(mcp_server_dir, "tool_get_price_alpaca.py"),
                "name": "AlpacaPrices",
                "port": self.ports["alpaca_price"],
            },
            "alpaca_trade": {
                "script": os.path.join(mcp_server_dir, "tool_trade_alpaca.py"),
                "name": "AlpacaTrade",
                "port": self.ports["alpaca_trade"],
            },
            "alpaca_news": {
                "script": os.path.join(mcp_server_dir, "tool_news_alpaca.py"),
                "name": "AlpacaNews",
                "port": self.ports["alpaca_news"],
            },
            "trade_flow": {
                "script": os.path.join(mcp_server_dir, "tool_trade_flow.py"),
                "name": "TradeFlow",
                "port": self.ports["trade_flow"],
            },
            "corporate_actions": {
                "script": os.path.join(mcp_server_dir, "tool_corporate_actions.py"),
                "name": "CorporateActions",
                "port": self.ports["corporate_actions"],
            },
            "onchain": {
                "script": os.path.join(mcp_server_dir, "tool_onchain_metrics.py"),
                "name": "OnChainMetrics",
                "port": self.ports["onchain"],
            },
        }

        # Create logs directory
        self.log_dir = Path(__file__).parent.parent / "logs" / "alpaca"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print("\n🛑 Received stop signal, shutting down all services...")
        self.stop_all_services()
        sys.exit(0)

    def is_port_available(self, port):
        """Check if a port is available"""
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result != 0
        except Exception:
            return False

    def check_alpaca_credentials(self):
        """Verify Alpaca API credentials are configured"""
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            print("❌ Alpaca API credentials not found!")
            print("   Please set the following environment variables:")
            print("   - ALPACA_API_KEY")
            print("   - ALPACA_SECRET_KEY")
            print("\n   You can get these from https://alpaca.markets/")
            return False

        print("✅ Alpaca credentials found")
        return True

    def check_port_conflicts(self, auto_find_ports=False):
        """Check for port conflicts before starting services"""
        conflicts = []
        for service_id, config in self.service_configs.items():
            port = config["port"]
            if not self.is_port_available(port):
                conflicts.append((config["name"], port))

        if conflicts:
            print("⚠️  Port conflicts detected:")
            for name, port in conflicts:
                print(f"   - {name}: Port {port} is already in use")

            # Auto-find ports if flag is set or if running non-interactively
            if auto_find_ports or not sys.stdin.isatty():
                print("\n🔄 Automatically finding available ports...")
                for service_id, config in self.service_configs.items():
                    port = config["port"]
                    if not self.is_port_available(port):
                        new_port = port
                        while not self.is_port_available(new_port):
                            new_port += 1
                            if new_port > port + 100:
                                print(f"❌ Could not find available port for {config['name']}")
                                return False
                        print(f"   ✅ {config['name']}: Changed port from {port} to {new_port}")
                        config["port"] = new_port
                        self.ports[service_id] = new_port
                return True

            response = input("\n❓ Do you want to automatically find available ports? (y/n): ")
            if response.lower() == "y":
                for service_id, config in self.service_configs.items():
                    port = config["port"]
                    if not self.is_port_available(port):
                        new_port = port
                        while not self.is_port_available(new_port):
                            new_port += 1
                            if new_port > port + 100:
                                print(f"❌ Could not find available port for {config['name']}")
                                return False
                        print(f"   ✅ {config['name']}: Changed port from {port} to {new_port}")
                        config["port"] = new_port
                        self.ports[service_id] = new_port
                return True
            else:
                print("\n💡 Tip: Stop the conflicting services or change port configuration")
                return False
        return True

    def start_service(self, service_id, config):
        """Start a single service"""
        script_path = config["script"]
        service_name = config["name"]
        port = config["port"]

        if not Path(script_path).exists():
            print(f"❌ Script file not found: {script_path}")
            return False

        try:
            # Set port environment variable
            env = os.environ.copy()
            port_env_map = {
                "alpaca_price": "ALPACA_PRICE_PORT",
                "alpaca_trade": "ALPACA_TRADE_PORT",
                "alpaca_news": "ALPACA_NEWS_PORT",
                "trade_flow": "TRADE_FLOW_PORT",
                "corporate_actions": "CORPORATE_ACTIONS_PORT",
                "onchain": "ONCHAIN_PORT",
            }
            if service_id in port_env_map:
                env[port_env_map[service_id]] = str(port)

            # Start service process
            log_file = self.log_dir / f"{service_id}.log"
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                    env=env,
                )

            self.services[service_id] = {
                "process": process,
                "name": service_name,
                "port": port,
                "log_file": log_file,
            }

            print(f"✅ {service_name} service started (PID: {process.pid}, Port: {port})")
            return True

        except Exception as e:
            print(f"❌ Failed to start {service_name} service: {e}")
            return False

    def check_service_health(self, service_id):
        """Check service health status"""
        if service_id not in self.services:
            return False

        service = self.services[service_id]
        process = service["process"]
        port = service["port"]

        # Check if process is still running
        if process.poll() is not None:
            return False

        # Check if port is responding
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def start_all_services(self):
        """Start all services"""
        print("🚀 Starting Alpaca MCP services...")
        print("=" * 50)

        # Check Alpaca credentials first
        if not self.check_alpaca_credentials():
            print("\n❌ Cannot start services without Alpaca credentials")
            return False

        # Check for port conflicts
        if not self.check_port_conflicts():
            print("\n❌ Cannot start services due to port conflicts")
            return False

        print(f"\n📊 Port configuration:")
        for service_id, config in self.service_configs.items():
            print(f"  - {config['name']}: {config['port']}")

        print("\n🔄 Starting services...")

        # Start all services
        success_count = 0
        for service_id, config in self.service_configs.items():
            if self.start_service(service_id, config):
                success_count += 1

        if success_count == 0:
            print("\n❌ No services started successfully")
            return False

        # Wait for services to start
        print("\n⏳ Waiting for services to start...")
        time.sleep(3)

        # Check service status
        print("\n🔍 Checking service status...")
        healthy_count = self.check_all_services()

        if healthy_count > 0:
            print(f"\n🎉 {healthy_count}/{len(self.services)} Alpaca MCP services running!")
            self.print_service_info()
            self.keep_alive()
            return True
        else:
            print("\n❌ All services failed to start properly")
            self.stop_all_services()
            return False

    def check_all_services(self):
        """Check all service status and return count of healthy services"""
        healthy_count = 0
        for service_id, service in self.services.items():
            if self.check_service_health(service_id):
                print(f"✅ {service['name']} service running normally")
                healthy_count += 1
            else:
                print(f"❌ {service['name']} service failed to start")
                print(f"   Check logs: {service['log_file']}")
        return healthy_count

    def print_service_info(self):
        """Print service information"""
        print("\n📋 Service information:")
        for service_id, service in self.services.items():
            print(f"  - {service['name']}: http://localhost:{service['port']} (PID: {service['process'].pid})")

        print(f"\n📁 Log files location: {self.log_dir.absolute()}")
        print("\n🛑 Press Ctrl+C to stop all services")

    def keep_alive(self):
        """Keep services running"""
        try:
            while self.running:
                time.sleep(5)

                # Check service status
                stopped_services = []
                for service_id, service in self.services.items():
                    if service["process"].poll() is not None:
                        stopped_services.append(service["name"])

                if stopped_services:
                    print(f"\n⚠️  Following service(s) stopped unexpectedly: {', '.join(stopped_services)}")
                    print(f"📋 Active services: {len(self.services) - len(stopped_services)}/{len(self.services)}")

                    if len(stopped_services) == len(self.services):
                        print("❌ All services have stopped, shutting down...")
                        self.running = False
                        break

        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all_services()

    def stop_all_services(self):
        """Stop all services"""
        print("\n🛑 Stopping all services...")

        for service_id, service in self.services.items():
            try:
                service["process"].terminate()
                service["process"].wait(timeout=5)
                print(f"✅ {service['name']} service stopped")
            except subprocess.TimeoutExpired:
                service["process"].kill()
                print(f"🔨 {service['name']} service force stopped")
            except Exception as e:
                print(f"❌ Error stopping {service['name']} service: {e}")

        print("✅ All services stopped")

    def status(self):
        """Display service status"""
        print("📊 Alpaca MCP Service Status Check")
        print("=" * 40)

        for service_id, config in self.service_configs.items():
            if service_id in self.services:
                service = self.services[service_id]
                if self.check_service_health(service_id):
                    print(f"✅ {config['name']} service running normally (Port: {config['port']})")
                else:
                    print(f"❌ {config['name']} service abnormal (Port: {config['port']})")
            else:
                print(f"⚫ {config['name']} service not started (Port: {config['port']})")


def main():
    """Main function"""
    print("=" * 50)
    print("  Alpaca Paper Trading MCP Services")
    print("=" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        manager = AlpacaMCPServiceManager()
        manager.status()
    else:
        manager = AlpacaMCPServiceManager()
        manager.start_all_services()


if __name__ == "__main__":
    main()
