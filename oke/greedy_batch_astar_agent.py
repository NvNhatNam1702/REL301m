import numpy as np
import itertools
from warehouse_env_multipod import WarehouseEnvMultiPod
import heapq

class AStarAgent:
    def __init__(self, env):
        self.env = env
        self.grid_height = env.grid_height
        self.grid_width = env.grid_width

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos):
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_r, new_c = pos[0] + dr, pos[1] + dc
            if (0 <= new_r < self.grid_height and 0 <= new_c < self.grid_width):
                neighbors.append((new_r, new_c))
        return neighbors

    def find_path(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        while frontier:
            current = heapq.heappop(frontier)[1]
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            for next_pos in self.get_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(goal, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        return None

class GreedyBatchAStarAgent:
    def __init__(self, env, batch_size=3):
        self.env = env
        self.astar = AStarAgent(env)
        self.batch_size = batch_size
        self.delivery = env.delivery_locations[0]

    def get_pod_cells(self):
        return [(r, c, self.env.grid[r, c]) for r in range(self.env.grid_height) 
                for c in range(self.env.grid_width) if self.env.grid[r, c] > 0]

    def select_greedy_batch(self, start):
        pod_cells = self.get_pod_cells()
        expanded = []
        for r, c, count in pod_cells:
            expanded.extend([(r, c)] * count)
        best_batch = None
        best_sum = 0
        for k in range(self.batch_size, 0, -1):
            for combo in itertools.combinations(expanded, k):
                unique_cells = list(dict.fromkeys(combo))
                pod_sum = len(combo)
                if pod_sum > 3:
                    continue
                if pod_sum > best_sum:
                    best_sum = pod_sum
                    best_batch = unique_cells
            if best_batch:
                break
        return best_batch if best_batch else []

    def plan_batch_path(self, start, batch):
        best_path = None
        best_order = None
        min_length = float('inf')
        for perm in itertools.permutations(batch):
            order = list(perm)
            current = start
            full_path = []
            valid = True
            for cell in order:
                segment = self.astar.find_path(current, cell)
                if segment is None:
                    valid = False
                    break
                if full_path:
                    full_path += segment[1:]
                else:
                    full_path += segment
                current = cell
            if not valid:
                continue
            segment = self.astar.find_path(current, self.delivery)
            if segment is None:
                continue
            full_path += segment[1:]
            if len(full_path) < min_length:
                min_length = len(full_path)
                best_path = full_path
                best_order = order
        if best_path is None:
            return None
        return best_path, best_order

    def run(self):
        env = self.env
        total_steps = 0
        while not env.all_pods_delivered():
            start = env.robot_pos
            batch = self.select_greedy_batch(start)
            if not batch:
                print("No more pods to pick up!")
                break
            plan = self.plan_batch_path(start, batch)
            if plan is None:
                print("No path found for batch", batch)
                break
            path, order = plan
            print(f"Batch: {order}, path length: {len(path)}")
            for pos in path:
                env.update_robot_position(pos)
                env.render()
                if pos in order and env.grid[pos] > 0:
                    env.mark_pod_delivered(pos)
                    print(f"Delivered pod at {pos}, pods left: {np.sum(env.grid)}")
            env.robot_pos = self.delivery
            total_steps += len(path)
        print(f"All pods delivered! Total steps: {total_steps}")
        env.window.mainloop()

if __name__ == "__main__":
    env = WarehouseEnvMultiPod()
    agent = GreedyBatchAStarAgent(env)
    agent.run()