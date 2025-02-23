# import gurobipy as gp
# try:
#     m = gp.Model("test")
#     print("Gurobi license is working!")
# except gp.GurobiError as e:
#     print(f"Error: {e}")

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt


def define_problem_data():
    """
    Define the data required for the system model:
    - L: Set of resolutions
    - B_l: Bandwidth demand for each resolution
    - Q_l: Quality score for each resolution
    - L_u: Set of resolutions selectable by users (dict)
    - C_{i,j}: Capacity of link (i->j)
    - w_u: User weights
    """

    # Resolution bandwidth demand (Mbps)
    B_l = {
        '8K': 200,
        '4K': 45,
        '2K': 16,
        '1K': 8
    }

    # Resolution quality scores
    Q_l = {
        '8K': 4,
        '4K': 3,
        '2K': 2,
        '1K': 1
    }

    # Set of resolutions selectable by users L_u
    L_u = {
        'A': ['1K'],
        'B': ['1K', '2K', '4K'],
        'C': ['1K', '2K', '4K', '8K'],
        'D': ['1K', '2K', '4K', '8K'],
        'E': ['1K', '2K'],
        'F': ['1K', '2K', '4K'],
        'G': ['1K', '2K', '4K', '8K'],
        'H': ['1K', '2K', '4K', '8K'],
        'I': ['1K', '2K', '4K'],
        'J': ['1K', '2K', '4K']
    }

    # Link capacity C_{i,j} (i->j)
    C_ij = {
        ('Server', 'Core_Forwarder_1'): 500,
        ('Server', 'Core_Forwarder_2'): 500,
        ('Core_Forwarder_1', 'Edge_Forwarder_A'): 300,
        ('Core_Forwarder_1', 'Edge_Forwarder_B'): 300,
        ('Core_Forwarder_2', 'Edge_Forwarder_A'): 200,
        ('Core_Forwarder_2', 'Edge_Forwarder_B'): 300,
        ('Core_Forwarder_2', 'Edge_Forwarder_C'): 400,
        ('Edge_Forwarder_A', 'Client_A'): 150,
        ('Edge_Forwarder_A', 'Client_B'): 150,
        ('Edge_Forwarder_B', 'Client_C'): 200,
        ('Edge_Forwarder_B', 'Client_D'): 200,
        ('Edge_Forwarder_C', 'Client_E'): 100,
        ('Edge_Forwarder_C', 'Client_F'): 100,
        ('Edge_Forwarder_C', 'Client_G'): 100,
        ('Edge_Forwarder_A', 'Client_H'): 100,
        ('Edge_Forwarder_B', 'Client_I'): 150,
        ('Edge_Forwarder_C', 'Client_J'): 200
    }

    # User weights w_u (default is 1)
    w_u = {
        'A': 1,
        'B': 1,
        'C': 1,
        'D': 1,
        'E': 1,
        'F': 1,
        'G': 1,
        'H': 1,
        'I': 1,
        'J': 1
    }

    return B_l, Q_l, L_u, C_ij, w_u


class ContinuousCentralizedModel:
    def __init__(self):
        # Load data
        self.B_l, self.Q_l, self.L_u, self.C_ij, self.w_u = define_problem_data()

        # User set U (e.g., 'A', 'B', 'C'...)
        self.U = list(self.L_u.keys())
        # Resolution set L (e.g., '8K', '4K', '2K', '1K')
        self.L = list(self.B_l.keys())

        # Define server / forwarder / user node sets
        self.S = {'Server'}
        self.F = {'Core_Forwarder_1', 'Core_Forwarder_2',
                  'Edge_Forwarder_A', 'Edge_Forwarder_B', 'Edge_Forwarder_C'}
        # User nodes written as "Client_X"
        self.U_nodes = {f"Client_{u}" for u in self.U}

        # Entire node set V = S ∪ F ∪ U_nodes
        self.V = self.S | self.F | self.U_nodes

        # Link set E (i->j)
        self.E = list(self.C_ij.keys())

        # Prepare model and variables
        self.model = None
        self.x = None  # x[i,l] ∈ [0,1]
        self.y = None  # y[i,j,l] ∈ [0,1]
        self.b = None  # b[i,j] = ∑(l) B_l * y[i,j,l]

        self._create_model()

    def _create_model(self):
        """Create Gurobi model and variables (continuous in [0,1])"""
        self.model = gp.Model("ContinuousCentralizedModel")

        # x[i,l] ∈ [0,1]
        self.x = self.model.addVars(
            self.V, self.L,
            vtype=GRB.BINARY,
            name="x"
        )

        # y[i,j,l] ∈ [0,1]
        self.y = self.model.addVars(
            self.E, self.L,
            vtype=GRB.BINARY,
            name="y"
        )

        # b[i,j] = actual bandwidth usage of link (i->j)
        self.b = self.model.addVars(
            self.E,
            vtype=GRB.CONTINUOUS,
            name="b"
        )

    def set_objective(self):
        """Objective function: max ∑(u in U) w_u * ∑(l in L) Q_l * x[u,l]."""
        obj_expr = gp.quicksum(
            self.w_u[u] * self.Q_l[l] * self.x[f"Client_{u}", l]
            for u in self.U
            for l in self.L
        )
        self.model.setObjective(obj_expr, GRB.MAXIMIZE)

    def add_constraints(self):
        """Add all linear constraints corresponding to the system model."""
        # 1) Server nodes possess all resolutions: x[s,l] = 1
        for s in self.S:
            for l in self.L:
                self.model.addConstr(
                    self.x[s, l] == 1,
                    name=f"server_owns_{s}_{l}"
                )

        # 2a) User capability limit: if l is not in L_u[u], then x[u,l] = 0
        for u in self.U:
            allowed_set = set(self.L_u[u])  # Resolutions selectable by this user
            for l in self.L:
                if l not in allowed_set:
                    self.model.addConstr(
                        self.x[f"Client_{u}", l] == 0,
                        name=f"user_cap_{u}_{l}"
                    )

        # 2b) Each user selects at most 1 resolution: ∑(l in L) x[u,l] ≤ 1
        for u in self.U:
            self.model.addConstr(
                gp.quicksum(self.x[f"Client_{u}", l] for l in self.L) <= 1,
                name=f"user_single_res_{u}"
            )

        # 3a) Transmission logic: y[i,j,l] ≤ x[i,l]
        for (i, j) in self.E:
            for l in self.L:
                self.model.addConstr(
                    self.y[i, j, l] <= self.x[i, l],
                    name=f"trans_logic_1_{i}_{j}_{l}"
                )

        # 3b) Non-server nodes need at least one upstream transmission to possess l: x[i,l] ≤ ∑(p->i) y[p,i,l]
        for node in (self.F | self.U_nodes):
            in_edges = [(p, node) for (p, q) in self.E if q == node]
            for l in self.L:
                self.model.addConstr(
                    self.x[node, l] <= gp.quicksum(self.y[p, node, l] for (p, _) in in_edges),
                    name=f"trans_logic_2_{node}_{l}"
                )

        # 4) Bandwidth capacity: ∑(l) B_l * y[i,j,l] ≤ C_{i,j}
        for (i, j), cap in self.C_ij.items():
            self.model.addConstr(
                gp.quicksum(self.B_l[l] * self.y[i, j, l] for l in self.L) <= cap,
                name=f"capacity_{i}_{j}"
            )

        # 5) b[i,j] = ∑(l) B_l * y[i,j,l]
        for (i, j) in self.E:
            self.model.addConstr(
                self.b[i, j] == gp.quicksum(self.B_l[l] * self.y[i, j, l] for l in self.L),
                name=f"bandwidth_usage_{i}_{j}"
            )

    def optimize(self):
        """Perform optimization"""
        self.model.optimize()

    def print_results(self):
        """Print and visualize results"""
        if self.model.status == GRB.OPTIMAL:
            print("\nOptimal solution found!")
            print(f"Optimal Objective Value = {self.model.ObjVal}\n")

            # Check the values of x[u,l] for users
            results = []
            for u in self.U:
                for l in self.L:
                    val = self.x[f"Client_{u}", l].X
                    if val > 1e-6:
                        results.append({
                            'User': u,
                            'Resolution(l)': l,
                            'x[u,l]': round(val, 4),
                            'Bandwidth(B_l)': self.B_l[l]
                        })
            df_results = pd.DataFrame(results)
            print("User resolution selection (continuous x):")
            print(df_results)

            # Link bandwidth usage
            link_usage = []
            for (i, j) in self.E:
                used = self.b[i, j].X
                cap = self.C_ij[(i, j)]
                link_usage.append({
                    'From': i,
                    'To': j,
                    'Usage(Mbps)': round(used, 2),
                    'Capacity(Mbps)': cap,
                    'Utilization(%)': round(used / cap * 100, 2) if cap > 1e-9 else 0
                })
            df_links = pd.DataFrame(link_usage)
            print("\nLink bandwidth usage:")
            print(df_links)

            self.plot_network(df_links)
        else:
            print("No optimal solution found. Status:", self.model.status)

    def plot_network(self, df_links):
        fig, ax = plt.subplots(figsize=(15, 10))

        pos = {
            'Server': (0.2, 0.8),
            'Core_Forwarder_1': (0.4, 0.7),
            'Core_Forwarder_2': (0.6, 0.7),
            'Edge_Forwarder_A': (0.3, 0.5),
            'Edge_Forwarder_B': (0.5, 0.5),
            'Edge_Forwarder_C': (0.7, 0.5),
            'Client_A': (0.1, 0.3),
            'Client_B': (0.2, 0.3),
            'Client_C': (0.4, 0.3),
            'Client_D': (0.5, 0.3),
            'Client_E': (0.6, 0.3),
            'Client_F': (0.7, 0.3),
            'Client_G': (0.8, 0.3),
            'Client_H': (0.3, 0.3),
            'Client_I': (0.45, 0.2),
            'Client_J': (0.65, 0.2),
        }

        user_choices = {}
        for u in self.U:
            chosen_info = []
            for l in self.L:
                val = self.x[f"Client_{u}", l].X
                if val > 1e-6:
                    chosen_info.append(f"{l}({val:.2f})")
            user_choices[u] = chosen_info

        utilization_dict = {
            (row['From'], row['To']): row['Utilization(%)']
            for _, row in df_links.iterrows()
        }

        def get_color(util_pct):
            return plt.cm.RdYlGn_r(min(util_pct, 100) / 100.0)

        # Draw nodes
        for node, (x_c, y_c) in pos.items():
            if node.startswith("Client_"):
                ax.plot(x_c, y_c, 'gs', markersize=20)
                ax.text(x_c, y_c - 0.02, node, ha='center')

                uid = node.split('_')[1]
                allowed = self.L_u[uid]
                chosen = user_choices[uid]
                chosen_str = ", ".join(chosen) if chosen else "N/A"

                ax.text(
                    x_c, y_c - 0.05,
                    f"Allowed: {','.join(allowed)}\nChosen: {chosen_str}",
                    ha='center', fontsize=8
                )
            elif "Forwarder" in node:
                ax.plot(x_c, y_c, 'bo', markersize=20)
                ax.text(x_c, y_c - 0.02, node, ha='center')
            else:
                ax.plot(x_c, y_c, 'rd', markersize=20)
                ax.text(x_c, y_c - 0.02, node, ha='center')

        # Draw links with arrows
        for (i, j), util_pct in utilization_dict.items():
            if i not in pos or j not in pos:
                continue
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            
            # Calculate arrow position (70% along the line)
            arrow_pos = 0.7
            arrow_x = x1 + arrow_pos * (x2 - x1)
            arrow_y = y1 + arrow_pos * (y2 - y1)
            
            # Draw line
            ax.plot([x1, x2], [y1, y2], '-', color=get_color(util_pct), linewidth=2)
            
            # Add arrow
            dx = x2 - x1
            dy = y2 - y1
            ax.arrow(arrow_x, arrow_y, dx*0.15, dy*0.15, 
                    head_width=0.02, head_length=0.02, 
                    fc=get_color(util_pct), ec=get_color(util_pct))

            cap_ij = self.C_ij[(i, j)]
            label = f"{round(util_pct,1)}%\n({cap_ij} Mbps)"
            
            # Adjust label positions for specific edges to avoid overlap
            if (i, j) == ('Core_Forwarder_1', 'Edge_Forwarder_B'):
                label_offset = -0.03  # Move label down
            elif (i, j) == ('Core_Forwarder_2', 'Edge_Forwarder_A'):
                label_offset = 0.05   # Move label up
            else:
                label_offset = 0.02   # Default offset
                
            ax.text((x1 + x2)/2, (y1 + y2)/2 + label_offset, label)

        # Color bar
        norm = plt.Normalize(0, 100)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Link Utilization(%)')

        # Resolution legend
        ax.text(0.2, 0.85, 'Resolutions:', ha='left')
        for idx, l in enumerate(self.L):
            ax.text(0.2, 0.83 - idx * 0.02, f"{l} -- {self.B_l[l]} Mbps", ha='left')

        ax.set_title('Centralized Model: Network Topology and Link Utilization')
        ax.axis('off')
        plt.tight_layout()
        plt.show()


def run_optimization():
    model = ContinuousCentralizedModel()
    model.set_objective()
    model.add_constraints()
    model.optimize()
    model.print_results()


if __name__ == "__main__":
    run_optimization()
