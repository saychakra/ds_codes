# K-Nearest Neighbors (KNN)

## What does this code do?
Implements the KNN algorithm from scratch. KNN is a simple, instance-based learning algorithm that classifies new points based on the majority class of their k nearest neighbors in the training data.

## Key Components
- **Distance Metrics**: Euclidean distance calculation between points
- **Neighbor Search**: Finding k closest training samples
- **Voting Mechanism**: Majority vote among neighbors
- **Prediction**: Classification based on neighbor votes

## Learning Concepts
- Lazy learning vs eager learning
- Distance metrics (Euclidean, Manhattan, etc.)
- Curse of dimensionality
- Effect of k parameter on model behavior

## Algorithm Flow
```
New point → Calculate distances to all training points 
→ Sort by distance → Select k nearest → Majority vote 
→ Return predicted class
```

## Pros & Cons
✅ Simple to understand and implement
✅ No training phase required
❌ Slow prediction time (O(n))
❌ Memory intensive
❌ Sensitive to irrelevant features

## Related Files
- `classification_evaluation_metrics_scratch.ipynb` - Evaluate KNN performance
- `knn.py` - Alternative implementation

## Further Reading
- Understanding decision boundaries in KNN
- How to choose optimal k value
- Feature scaling importance for distance-based algorithms
