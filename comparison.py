import time
import tkinter as tk
from warehouse_env_astar import WarehouseEnvAStar
from hybrid_warehouse_agent import HybridWarehouseAgent
from metaheuristic_path_planner import find_optimal_batch_and_order
from agent_astar import AStarAgent

class ComparisonWindow:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Warehouse Algorithms Comparison")
        
        # Create two frames side by side
        self.left_frame = tk.Frame(self.window)
        self.right_frame = tk.Frame(self.window)
        self.left_frame.pack(side=tk.LEFT, padx=10)
        self.right_frame.pack(side=tk.LEFT, padx=10)
        
        # Create environments
        self.env_hybrid = WarehouseEnvAStar()
        self.env_meta = WarehouseEnvAStar()
        
        # Create agents
        self.hybrid_agent = HybridWarehouseAgent(
            self.env_hybrid,
            batch_size=3,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size_dqn=32,
            target_update=10
        )
        self.meta_agent = AStarAgent(self.env_meta)
        
        # Metrics
        self.hybrid_metrics = {
            'total_distance': 0,
            'total_time': 0,
            'pods_delivered': 0
        }
        self.meta_metrics = {
            'total_distance': 0,
            'total_time': 0,
            'pods_delivered': 0
        }
        
        # Create labels for metrics
        self.create_metric_labels()
        
        # Start the comparison
        self.run_comparison()
        
    def create_metric_labels(self):
        # Hybrid metrics
        tk.Label(self.left_frame, text="Hybrid DQN Agent", font=('Arial', 14, 'bold')).pack()
        self.hybrid_distance_label = tk.Label(self.left_frame, text="Total Distance: 0")
        self.hybrid_time_label = tk.Label(self.left_frame, text="Total Time: 0.0s")
        self.hybrid_pods_label = tk.Label(self.left_frame, text="Pods Delivered: 0")
        self.hybrid_distance_label.pack()
        self.hybrid_time_label.pack()
        self.hybrid_pods_label.pack()
        
        # Meta metrics
        tk.Label(self.right_frame, text="Metaheuristic Agent", font=('Arial', 14, 'bold')).pack()
        self.meta_distance_label = tk.Label(self.right_frame, text="Total Distance: 0")
        self.meta_time_label = tk.Label(self.right_frame, text="Total Time: 0.0s")
        self.meta_pods_label = tk.Label(self.right_frame, text="Pods Delivered: 0")
        self.meta_distance_label.pack()
        self.meta_time_label.pack()
        self.meta_pods_label.pack()
    
    def update_metrics(self):
        # Update hybrid metrics
        self.hybrid_distance_label.config(text=f"Total Distance: {self.hybrid_metrics['total_distance']}")
        self.hybrid_time_label.config(text=f"Total Time: {self.hybrid_metrics['total_time']:.2f}s")
        self.hybrid_pods_label.config(text=f"Pods Delivered: {self.hybrid_metrics['pods_delivered']}")
        
        # Update meta metrics
        self.meta_distance_label.config(text=f"Total Distance: {self.meta_metrics['total_distance']}")
        self.meta_time_label.config(text=f"Total Time: {self.meta_metrics['total_time']:.2f}s")
        self.meta_pods_label.config(text=f"Pods Delivered: {self.meta_metrics['pods_delivered']}")
    
    def run_hybrid_agent(self):
        start_time = time.time()
        while self.env_hybrid.available_pods:
            success = self.hybrid_agent.process_batch()
            if not success:
                break
            # Count actual pods delivered in this batch
            pods_delivered = len(self.hybrid_agent.batch_processor.select_pods_for_batch(
                self.env_hybrid.available_pods, self.env_hybrid.robot_pos))
            self.hybrid_metrics['pods_delivered'] += pods_delivered
            self.hybrid_metrics['total_distance'] += len(self.env_hybrid.path)
            self.update_metrics()
            self.window.update()
        self.hybrid_metrics['total_time'] = time.time() - start_time
        self.update_metrics()
    
    def run_meta_agent(self):
        start_time = time.time()
        while self.env_meta.inventory_pods:
            batch, order, cost = find_optimal_batch_and_order(
                self.env_meta.inventory_pods,
                self.env_meta.robot_pos,
                self.env_meta.delivery_locations[0],
                self.meta_agent,
                3
            )
            if not batch:
                break
                
            current = self.env_meta.robot_pos
            for pod in order:
                self.env_meta.current_target_pod = pod
                self.env_meta.target_pod = pod
                segment = self.meta_agent.find_path(current, pod)
                if segment:
                    self.meta_metrics['total_distance'] += len(segment) - 1
                    for pos in segment[:-1]:
                        self.env_meta.update_robot_position(pos)
                        self.env_meta.render()
                    if pod in self.env_meta.inventory_pods:
                        self.env_meta.inventory_pods.remove(pod)
                        self.env_meta.grid[pod] = 0
                    current = pod
                self.env_meta.current_target_pod = None
                self.env_meta.target_pod = None
                self.env_meta.render()
                self.update_metrics()
                self.window.update()
            
            segment = self.meta_agent.find_path(current, self.env_meta.delivery_locations[0])
            if segment:
                self.meta_metrics['total_distance'] += len(segment) - 1
                for pos in segment:
                    self.env_meta.update_robot_position(pos)
                    self.env_meta.target_pod = None
                    self.env_meta.render()
                self.meta_metrics['pods_delivered'] += len(batch)
                self.update_metrics()
                self.window.update()
        
        self.meta_metrics['total_time'] = time.time() - start_time
        self.update_metrics()
    
    def run_comparison(self):
        # Run both agents in parallel
        self.run_hybrid_agent()
        self.run_meta_agent()
        
        # Print final comparison
        print("\nFinal Comparison:")
        print("Hybrid DQN Agent:")
        print(f"Total Distance: {self.hybrid_metrics['total_distance']}")
        print(f"Total Time: {self.hybrid_metrics['total_time']:.2f}s")
        print(f"Pods Delivered: {self.hybrid_metrics['pods_delivered']}")
        print("\nMetaheuristic Agent:")
        print(f"Total Distance: {self.meta_metrics['total_distance']}")
        print(f"Total Time: {self.meta_metrics['total_time']:.2f}s")
        print(f"Pods Delivered: {self.meta_metrics['pods_delivered']}")
        
        self.window.mainloop()

if __name__ == "__main__":
    comparison = ComparisonWindow() 