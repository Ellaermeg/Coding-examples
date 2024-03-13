from Map import Map_Obj
import os
import heapq
import numpy


def main():

##### INSTRUCTIONS FOR THE ALGORTIHM #####

# In order to change which task the algorithm should tackle, simply change the task number in the map_obj function on line 21



    M = Map_Obj()
    print("--" * 10 + "starting" + "--" * 10) # Saftey check so see if main was being called
    current_dir = os.getcwd()
    
    
    #############################################################################################################
    map_obj = M.fill_critical_positions(task=4) # Specifies which map and also which start/goal positions to use
    #############################################################################################################
    
    
    
    grid = M.get_maps()[0] #initalizes the grid to be used
    start_node = node(map_obj[0], cost=1) # Uses the map_obj function from the Map.py file to specify the correct starting position for each task in part 1
    goal_node = node(map_obj[1], cost=1) # same as for start, just with the goal instead
    # Calculates cost based on the other arrays
    cost_map =  M.map_array(map_obj[3]) # creates a cost map to visualize how the cost affects the paths chosen
    
    
    
    ########## Quick note on the gird #########
    
    # In the png. which is generated, the path can bee seen divering into the separate ways, untill
    # it eventually finds the right path. Here you can see that it reaches a certain step, and for each
    # step it calculates the heuristic cost and compares it to the other states. 
    # For example, it goes all the way down to the "bodega", but then due to the hueristic cost increasing
    # the if statement containing the heapq algorithm makes sure that it checks to see if there are other states
    # which have a lower heuristic cost.
    
    path, visited_cells= Astar(grid, start_node, goal_node, cost_map)  # Calls the main Astar function as well as the visualization of the grid.
    if path:
        print("Path found:", path)
    else:
        print("No path found")
    
    visualize_path(grid, path, M)    
    visualize_path(grid, visited_cells, M)
    print(map_obj[3])
    print(M.get_maps())
    

class node:
    def __init__(self, state, parent=None, cost=1):
        self.state = state  # State of the current node which is searching. Looks for the completion of the algorithm, if the state is complete or not
        self.parent = parent  # The parents of the current node
        self.cost = cost  # The path cost to reach the current node
        
    def __lt__(self, other):
        return self.cost < other.cost # Define the less than comparison based on the cost

def Heuristic_cost(node, goal): # Estimates the given cost from the current node to the goal node
    x1, y1 = node.state
    #print("i am calculating EOEOEOEO")
    x2, y2 = goal.state
    #print("i am calculating BØØ")
    return abs(x1 - x2) + abs(y1 - y2) # Uses the manhatten distance to calulate the heuristic cost by using the absolute value of the node state


def Astar(grid, start, goal, cost_map):
    priority_queue_list = []  # Initially empty
    visited_nodes = [] # ques all the visited nodes once they have been added to priority list and removed
    
    heapq.heappush(priority_queue_list, (start.cost, start)) # Nodes are explored based on increasing total cost
    
    while priority_queue_list: 
        # Continues until the open list is unempty.
        # Nodes are explored based on increasing total cost
        current_cost, current_node = heapq.heappop(priority_queue_list) 
        if list(current_node.state) == list(goal.state): # Converts both properties into lists in order for them to be equal
            # Goal reached and the algorithm is complete
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            return path[::-1], visited_nodes
        visited_nodes.append(current_node.state)
        
        
        # Checks for if the goal has been reached. If so it doubles back through its parent nodes to reconstruct the path
        # Returns the result is the goal has been reached
        for neighbor in find_neighbors(grid, current_node, cost_map):
            if neighbor.state in visited_nodes:
                continue
        # Checks if the current node already has been visited
        # If this is the case the algorithm skips iterating through the nodes and costs and moves on to the next node
            n_cost = current_node.cost + 1
            if neighbor not in priority_queue_list:
            # If the current goal is neither visited or the goal it is added to the visited set of nodes
            # Iterates over the neighbors of "current_node" so that it can calculate the cost for each neighbor state.
                heapq.heappush(priority_queue_list, (n_cost + Heuristic_cost(neighbor, goal), neighbor))
                #print("CALCULATING")
            elif n_cost < neighbor.cost:
                neighbor.cost = n_cost
                neighbor.parent = current_node
    # Checks if there are any valid paths which can be found.            
    return None, visited_nodes
                
def find_neighbors(grid, current_node, cost_map):
    row, col = current_node.state # The specific location of the grid in rows and columns
    neighbors = [] # Stores the succesor states which will be reachable from the current state/node
    movement = [(1, 0), (0, 1), (-1, 0), (0, -1)] # specifies allowed movement 
    for dx, dy in movement:
        # For loop which calculates the new row and column for every new state, which occurs everytime there is a movement
        # Makes sure that there are noe illegal moves, such as stepping out of the provided grid
        new_row, new_col = row + dx, col + dy
        if(grid[new_row][new_col] == 1):
            cost = cost_map[new_row][new_col]
            neighbors.append(node((new_row, new_col), parent=current_node, cost=current_node.cost + cost))
        # Returns the list of valid neighbor states which can be reach from the different state input
    return neighbors



def visualize_path(grid, path, M): # Visualizes the grid using the implemented show_map function from the supplementary code
    for node in path: # Uses the path list, which contains all the appended nodes in the found path to show the quickest route from goal to start
        M.set_cell_value([node[0],node[1]], 'Z') # Here no specific reason for choosing Z exept that it was easier to look at in the string gird
        if path !=None:
            visited_nodes = path
    M.show_map()


if __name__ == "__main__":
    main()
