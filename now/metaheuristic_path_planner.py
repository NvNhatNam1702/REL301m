import random
import math
import itertools
from warehouse_env_astar import WarehouseEnvAStar
from agent_astar import AStarAgent

def total_path_length(order, start, delivery, agent):
    current = start
    total = 0
    for pod in order:
        # Set the pod as the target so the agent can step on it
        agent.env.target_pod = pod
        path = agent.find_path(current, pod)
        if not path:
            return float('inf')  # unreachable
        total += len(path) - 1
        current = pod
    # After last pod, go to delivery
    agent.env.target_pod = None
    path = agent.find_path(current, delivery)
    if not path:
        return float('inf')
    total += len(path) - 1
    return total

def find_optimal_batch_and_order(remaining_pods, start, delivery, agent, batch_size):
    best_batch = None
    best_order = None
    best_cost = float('inf')
    # Try all batch sizes from batch_size down to 1 (if fewer pods remain)
    for k in range(min(batch_size, len(remaining_pods)), 0, -1):
        for batch in itertools.combinations(remaining_pods, k):
            for order in itertools.permutations(batch):
                cost = total_path_length(order, start, delivery, agent)
                if cost < best_cost:
                    best_cost = cost
                    best_batch = batch
                    best_order = order
        if best_batch is not None:
            break  # Prefer largest possible batch
    return best_batch, best_order, best_cost

def main():
    env = WarehouseEnvAStar()
    agent = AStarAgent(env)
    batch_size = 3  # Configurable: max number of pods to pick up per delivery
    start = env.robot_pos
    delivery = env.delivery_locations[0]

    while env.inventory_pods:
        # Find the optimal batch and order
        batch, order, cost = find_optimal_batch_and_order(env.inventory_pods, start, delivery, agent, batch_size)
        if not batch:
            print("No reachable pods left!")
            break
        print(f"Next optimal batch: {order}, path length: {cost}")

        current = start
        for pod in order:
            # Highlight the current target pod in orange
            env.current_target_pod = pod
            env.target_pod = pod  # Ensure agent can step on pod
            segment = agent.find_path(current, pod)
            if segment:
                for pos in segment[:-1]:  # move to pod
                    env.update_robot_position(pos)
                    env.render()
                # Arrived at pod, color it white (remove from inventory)
                if pod in env.inventory_pods:
                    env.inventory_pods.remove(pod)
                    env.grid[pod] = 0
                current = pod
            # Remove highlight after pickup
            env.current_target_pod = None
            env.target_pod = None
            env.render()
        # Move to delivery
        segment = agent.find_path(current, delivery)
        if segment:
            for pos in segment:
                env.update_robot_position(pos)
                env.target_pod = None
                env.render()
            start = delivery  # Next batch starts from delivery location
    print("All pods have been picked up and delivered!")
    env.window.mainloop()

# Patch the environment's render method to support orange highlight for current_target_pod
orig_render = WarehouseEnvAStar.render
def patched_render(self):
    self.canvas.delete("all")
    for r in range(self.grid_height):
        for c in range(self.grid_width):
            x1 = c * self.cell_size
            y1 = r * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            if self.grid[r, c] == 1:
                if hasattr(self, 'current_target_pod') and self.current_target_pod == (r, c):
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="orange")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="green")
            elif (r, c) in self.workstations:
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="blue")
            elif (r, c) in self.delivery_locations:
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="yellow")
            else:
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")
    # Draw robot
    robot_x = self.robot_pos[1] * self.cell_size + self.cell_size // 2
    robot_y = self.robot_pos[0] * self.cell_size + self.cell_size // 2
    self.canvas.create_oval(
        robot_x - self.cell_size//3,
        robot_y - self.cell_size//3,
        robot_x + self.cell_size//3,
        robot_y + self.cell_size//3,
        fill="red"
    )
    # Draw path (optional, not used in this script)
    if hasattr(self, 'path') and self.path:
        for pos in self.path:
            x = pos[1] * self.cell_size + self.cell_size // 2
            y = pos[0] * self.cell_size + self.cell_size // 2
            self.canvas.create_oval(
                x - self.cell_size//6,
                y - self.cell_size//6,
                x + self.cell_size//6,
                y + self.cell_size//6,
                fill="red"
            )
    self.window.update()
WarehouseEnvAStar.render = patched_render

if __name__ == "__main__":
    main() 