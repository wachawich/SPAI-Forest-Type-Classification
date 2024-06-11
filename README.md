# SPAI-Forest-Type-Classification
Super AI engineer ss4 level2

# Resource
- **LightGBM** <br>
- **XGBoots** <br>
- **Random Forest** <br>
- **LogisticRegression** <br>
- **Catboots** <br>
 - **Optuna**

# Pipeline1 -  5 majority vote

# Pipeline2 -  Catboot + Optuna 


# Result and Score

**The results of my experiment**

	

|        Use     |              feature          	 |  Public | Private |
|----------------|-----------------------------------|---------|---------|
|Pipeline1       | Use 26 feature in indexbase   	 | 0.67599 | 0.68543 |
|Pipeline2       | Use 26 feature in indexbase   	 | 0.67599 | 0.69223 |
|Pipeline2		 | Use 171 feature in indexbase	 	 | 0.68107 | 0.70943 |
|Pipeline2		 | Use Top5 best score of Pipeline2 for majority vote | `0.68362` | `0.71091` |

