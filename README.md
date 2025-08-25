# Supply Chain Optimization & Visualization  

## 📌 Project Overview  
This project focuses on **optimizing supply chain operations** by aligning product supply capacities with international demand while minimizing **total operational costs**. Using **linear programming (PuLP)**, the model determines optimal product allocations across countries, balancing shipping costs and unmet demand penalties.  

The system produces both **structured reports (CSV)** and **presentation-ready visualizations**, making it a complete **decision-support tool** for supply chain management.  

---

## 🚀 Key Features  
- **Optimization Engine**  
  - Uses **Linear Programming** to minimize total costs (shipping + unmet demand penalty).  
  - Allocates product shipments across multiple countries.  

- **Generated Reports (CSV)**  
  - `allocation_by_product_country.csv` → product-to-country shipments  
  - `supply_summary_by_product.csv` → supply utilization & capacity tracking  
  - `demand_summary_by_country.csv` → demand fulfillment, unmet demand, fulfillment %  
  - `optimization_summary.csv` → total cost breakdown & solver status  

- **Data Visualizations (Matplotlib & Seaborn)**  
  - Demand vs. Fulfillment comparisons  
  - Supply vs. Shipped capacity charts  
  - Cost distribution breakdown (shipped vs. penalty)  
  - Top 10 countries by demand & Top 10 products by utilization  
  - Heatmap of product-to-country allocations  

---

## 📊 Example Outputs  
- **Fulfillment % by Country** – Identifies which regions face high unmet demand.  
- **Supply Utilization by Product** – Highlights over/under-utilized resources.  
- **Cost Composition** – Shows proportion of shipping vs. penalty costs.  
- **Top 10 Analysis** – Focus on critical countries and products.  

---

## 🛠️ Tools & Technologies  
- **Python**  
- **PuLP** – Linear Programming Optimization  
- **Pandas** – Data handling & summaries  
- **Matplotlib / Seaborn** – Data Visualization  
