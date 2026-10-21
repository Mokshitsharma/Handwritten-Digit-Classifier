from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

digits = datasets.load_digits()

X = digits.data
y = digits.target

fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Digit: {label}')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel='rbf', gamma=0.01, C=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

sample_index = 5
sample_image = X_test[sample_index].reshape(1, -1)
predicted_label = model.predict(sample_image)

print(f"Predicted label for the test image: {predicted_label[0]}")

plt.imshow(digits.images[len(X_train) + sample_index], cmap=plt.cm.gray_r)
plt.title(f"Predicted: {predicted_label[0]}")
plt.axis('off')
plt.show()
