# project.py
import os
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Settings ----------------
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Penalty multiplier for unmet demand (large to discourage unmet demand)
UNMET_PENALTY_MULTIPLIER = 1000.0

# ---------------- Helpers ----------------
def clean_columns(df):
    """Strip whitespace, lowercase, replace spaces with underscores."""
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# ---------------- Load & Clean ----------------
orders_path = os.path.join(DATA_DIR, "orders_and_shipments.csv")
inventory_path = os.path.join(DATA_DIR, "inventory.csv")

orders = pd.read_csv(orders_path, low_memory=False)
inventory = pd.read_csv(inventory_path, low_memory=False)

orders = clean_columns(orders)
inventory = clean_columns(inventory)

# Inspect expected columns (common variations)
# We'll try to find the best matching column names
# For orders:
# Expect: product name, order quantity, customer country (or customer_country)
orders_cols = orders.columns.tolist()
# print("orders cols:", orders_cols)

# Map likely column names to our canonical names
def find_col(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None

prod_col = find_col(orders_cols, ["product_name", "product", "productname", "product_name"])
qty_col = find_col(orders_cols, ["order_quantity", "quantity", "order_qty", "orderquantity", "qty"])
cust_col = find_col(orders_cols, ["customer_country", "customer_country", "customercountry", "customer_country"])

if prod_col is None or qty_col is None or cust_col is None:
    raise Exception(f"Could not find required columns in orders.csv. Found columns: {orders_cols}\n"
                    f"Expected something like product_name, order_quantity, customer_country")

# For inventory:
inv_cols = inventory.columns.tolist()
prod_col_inv = find_col(inv_cols, ["product_name", "product", "productname"])
inv_qty_col = find_col(inv_cols, ["warehouse_inventory", "warehouse_inventory", "warehouseinventory", "inventory"])
inv_cost_col = find_col(inv_cols, ["inventory_cost_per_unit", "inventory_cost", "cost_per_unit", "unit_cost", "inventory_cost_per_unit"])

if prod_col_inv is None or inv_qty_col is None or inv_cost_col is None:
    raise Exception(f"Could not find required columns in inventory.csv. Found columns: {inv_cols}\n"
                    f"Expected product_name, warehouse_inventory, inventory_cost_per_unit")

# ---------------- Build Demand (product x country) ----------------
# Ensure quantity is numeric
orders[qty_col] = pd.to_numeric(orders[qty_col], errors="coerce").fillna(0)

demand_df = (
    orders
    .groupby([prod_col, cust_col], dropna=False)[qty_col]
    .sum()
    .reset_index()
    .rename(columns={prod_col: "product", cust_col: "country", qty_col: "demand"})
)

# Remove zero-demand rows if any
demand_df = demand_df[demand_df["demand"] > 0].copy()

# ---------------- Build Supply (product-level) ----------------
inventory[inv_qty_col] = pd.to_numeric(inventory[inv_qty_col], errors="coerce").fillna(0)
inventory[inv_cost_col] = pd.to_numeric(inventory[inv_cost_col], errors="coerce").fillna(0)

supply_df = (
    inventory
    .groupby(prod_col_inv, dropna=False)
    .agg({
        inv_qty_col: "sum",
        inv_cost_col: "mean"
    })
    .reset_index()
    .rename(columns={prod_col_inv: "product", inv_qty_col: "supply_capacity", inv_cost_col: "unit_cost"})
)

# ---------------- Align product lists ----------------
products = sorted(set(supply_df["product"].unique()).union(set(demand_df["product"].unique())))
countries = sorted(demand_df["country"].unique())

# Convert to dicts for quick lookup (default 0 if missing)
supply_capacity = {row["product"]: float(row["supply_capacity"]) for _, row in supply_df.iterrows()}
unit_cost = {row["product"]: float(row["unit_cost"]) for _, row in supply_df.iterrows()}

# Ensure every product has an entry in supply_capacity and unit_cost
for p in products:
    if p not in supply_capacity:
        supply_capacity[p] = 0.0
    if p not in unit_cost:
        # If unit cost missing, use average of known costs or a default
        known_costs = [v for v in unit_cost.values() if v > 0]
        unit_cost[p] = float(sum(known_costs) / len(known_costs)) if known_costs else 1.0

# demand dict for quick lookup (product,country) -> demand
demand = {}
for _, r in demand_df.iterrows():
    demand[(r["product"], r["country"])] = float(r["demand"])

# For products/countries with no explicit demand row, demand is 0
# Build list of all (product,country) pairs to consider
pairs = []
for p in products:
    for c in countries:
        pairs.append((p, c))

# ---------------- Build LP (with unmet-demand variables) ----------------
model = pulp.LpProblem("Supply_Allocation_with_Unmet", pulp.LpMinimize)

# Decision variables: shipped quantity per (product,country)
ship = pulp.LpVariable.dicts(
    "ship",
    ( (p,c) for p,c in pairs ),
    lowBound=0,
    cat="Continuous"
)

# Unmet demand variables for each (product,country)
unmet = pulp.LpVariable.dicts(
    "unmet",
    ( (p,c) for p,c in pairs ),
    lowBound=0,
    cat="Continuous"
)

# Objective: cost of shipped units (unit_cost * qty) + large penalty * unmet demand
# Choose a penalty relative to average unit cost
avg_unit_cost = sum(unit_cost.values()) / max(1, len(unit_cost))
penalty = UNMET_PENALTY_MULTIPLIER * max(1.0, avg_unit_cost)

model += pulp.lpSum(
    ship[(p,c)] * unit_cost.get(p, avg_unit_cost) + unmet[(p,c)] * penalty
    for p,c in pairs
), "Total_Cost_Including_Unmet_Penalties"

# Supply constraints: for each product, total shipped across countries <= supply_capacity
for p in products:
    model += (
        pulp.lpSum(ship[(p,c)] for c in countries) <= supply_capacity.get(p, 0.0),
        f"SupplyCapacity_{p}"
    )

# Demand balance constraints: for each product-country: shipped + unmet == demand
for p,c in pairs:
    d = demand.get((p,c), 0.0)
    model += (
        ship[(p,c)] + unmet[(p,c)] == d,
        f"DemandBalance_{p}_{c}"
    )

# ---------------- Solve ----------------
solver = pulp.PULP_CBC_CMD(msg=False)  # silent solver
result_status = model.solve(solver)

print("Solver status:", pulp.LpStatus[model.status])

# ---------------- Collect results ----------------
alloc_rows = []
for p,c in pairs:
    shipped_qty = ship[(p,c)].value() or 0.0
    unmet_qty = unmet[(p,c)].value() or 0.0
    d = demand.get((p,c), 0.0)
    if shipped_qty > 1e-9 or unmet_qty > 1e-9:
        alloc_rows.append({
            "product": p,
            "country": c,
            "demand": d,
            "shipped_qty": round(shipped_qty, 6),
            "unmet_qty": round(unmet_qty, 6),
            "unit_cost": unit_cost.get(p, avg_unit_cost),
            "cost_shipped": round(shipped_qty * unit_cost.get(p, avg_unit_cost), 4),
            "cost_unmet_penalty": round(unmet_qty * penalty, 4)
        })

allocation_df = pd.DataFrame(alloc_rows)
allocation_df.to_csv(os.path.join(RESULTS_DIR, "allocation_by_product_country.csv"), index=False)

# Supplier (product) summary: total shipped, utilization, capacity, unit cost
supplier_rows = []
for p in products:
    total_shipped = allocation_df[allocation_df["product"] == p]["shipped_qty"].sum()
    cap = supply_capacity.get(p, 0.0)
    supplier_rows.append({
        "product": p,
        "supply_capacity": cap,
        "total_shipped": round(total_shipped, 6),
        "utilization_pct": round((total_shipped / cap * 100) if cap>0 else 0.0, 4),
        "unit_cost": unit_cost.get(p, avg_unit_cost)
    })

supplier_summary_df = pd.DataFrame(supplier_rows)
supplier_summary_df.to_csv(os.path.join(RESULTS_DIR, "supply_summary_by_product.csv"), index=False)

# Country (demand) summary: total demand, total shipped, unmet
country_rows = []
for c in countries:
    total_demand = sum(demand.get((p,c), 0.0) for p in products)
    total_shipped = allocation_df[allocation_df["country"] == c]["shipped_qty"].sum()
    total_unmet = allocation_df[allocation_df["country"] == c]["unmet_qty"].sum()
    country_rows.append({
        "country": c,
        "total_demand": round(total_demand, 6),
        "total_shipped": round(total_shipped, 6),
        "total_unmet": round(total_unmet, 6),
        "fulfillment_pct": round((total_shipped / total_demand * 100) if total_demand>0 else 100.0, 4)
    })

country_summary_df = pd.DataFrame(country_rows)
country_summary_df.to_csv(os.path.join(RESULTS_DIR, "demand_summary_by_country.csv"), index=False)

# Cost summary
total_shipped_cost = allocation_df["cost_shipped"].sum() if not allocation_df.empty else 0.0
total_unmet_penalty = allocation_df["cost_unmet_penalty"].sum() if not allocation_df.empty else 0.0
total_cost = total_shipped_cost + total_unmet_penalty

summary = {
    "total_shipped_cost": total_shipped_cost,
    "total_unmet_penalty": total_unmet_penalty,
    "total_cost": total_cost,
    "solver_status": pulp.LpStatus[model.status]
}
summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(RESULTS_DIR, "optimization_summary.csv"), index=False)

# Print key results for quick verification
print("\n--- Key Results ---")
print("Total shipped cost:", total_shipped_cost)
print("Total unmet penalty cost:", total_unmet_penalty)
print("Total cost (objective):", total_cost)
print("Allocations saved to:", os.path.join(RESULTS_DIR, "allocation_by_product_country.csv"))
print("Supply summary saved to:", os.path.join(RESULTS_DIR, "supply_summary_by_product.csv"))
print("Country summary saved to:", os.path.join(RESULTS_DIR, "demand_summary_by_country.csv"))
print("Optimization summary saved to:", os.path.join(RESULTS_DIR, "optimization_summary.csv"))


# ============= PATHS =============
results_folder = "results"

# Make sure results folder exists
os.makedirs(results_folder, exist_ok=True)

# ============= VISUALIZATIONS =============

print("\n--- Generating Visualizations ---")

# ---------- DEMAND ----------
try:
    demand_df = pd.read_csv("results/demand_summary_by_country.csv")
    print("Demand summary columns:", list(demand_df.columns))

    # Normalize column names
    demand_df.columns = demand_df.columns.str.strip().str.lower()

    if {"country", "total_demand", "total_shipped"}.issubset(demand_df.columns):
        # Full plot
        demand_df.set_index("country")[["total_demand", "total_shipped"]].plot(
            kind="bar", figsize=(12, 6), alpha=0.8
        )
        plt.title("Demand vs Fulfillment (All Countries)")
        plt.ylabel("Quantity")
        plt.tight_layout()
        plt.savefig("results/demand_vs_fulfillment.png")
        plt.close()

        # Top 10 clean plot
        top_demand = demand_df.sort_values("total_demand", ascending=False).head(10)
        top_demand.set_index("country")[["total_demand", "total_shipped"]].plot(
            kind="bar", figsize=(10, 5), alpha=0.8
        )
        plt.title("Top 10 Countries by Demand vs Fulfillment")
        plt.ylabel("Quantity")
        plt.tight_layout()
        plt.savefig("results/demand_vs_fulfillment_top10.png")
        plt.close()

        print("✅ Demand visualizations saved.")
    else:
        print("⚠️ Could not find required columns in demand summary for plotting.")

except Exception as e:
    print("❌ Demand visualization error:", e)


# ---------- SUPPLY ----------
try:
    supply_df = pd.read_csv("results/supply_summary_by_product.csv")
    print("Supply summary columns:", list(supply_df.columns))

    supply_df.columns = supply_df.columns.str.strip().str.lower()

    if {"product", "supply_capacity", "total_shipped"}.issubset(supply_df.columns):
        # Full plot
        supply_df.set_index("product")[["supply_capacity", "total_shipped"]].plot(
            kind="bar", figsize=(12, 6), alpha=0.8
        )
        plt.title("Supply Capacity vs Shipped (All Products)")
        plt.ylabel("Quantity")
        plt.tight_layout()
        plt.savefig("results/supply_vs_shipped.png")
        plt.close()

        # Top 10 clean plot
        top_supply = supply_df.sort_values("supply_capacity", ascending=False).head(10)
        top_supply.set_index("product")[["supply_capacity", "total_shipped"]].plot(
            kind="bar", figsize=(10, 5), alpha=0.8
        )
        plt.title("Top 10 Products by Supply vs Shipped")
        plt.ylabel("Quantity")
        plt.tight_layout()
        plt.savefig("results/supply_vs_shipped_top10.png")
        plt.close()

        print("✅ Supply visualizations saved.")
    else:
        print("⚠️ Could not find required columns in supply summary for plotting.")

except Exception as e:
    print("❌ Supply visualization error:", e)


# ---------- OPTIMIZATION SUMMARY ----------
try:
    opt_df = pd.read_csv("results/optimization_summary.csv")
    print("Optimization summary columns:", list(opt_df.columns))

    opt_df.columns = opt_df.columns.str.strip().str.lower()

    if {"total_shipped_cost", "total_unmet_penalty", "total_cost"}.issubset(opt_df.columns):
        opt_df[["total_shipped_cost", "total_unmet_penalty"]].plot(
            kind="bar", figsize=(8, 5), alpha=0.8
        )
        plt.title("Cost Breakdown")
        plt.ylabel("Cost")
        plt.tight_layout()
        plt.savefig("results/cost_breakdown.png")
        plt.close()

        print("✅ Optimization summary visualization saved.")
    else:
        print("⚠️ Could not find required columns in optimization summary for plotting.")

except Exception as e:
    print("❌ Optimization visualization error:", e)