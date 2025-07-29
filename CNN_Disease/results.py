import pickle
import matplotlib.pyplot as plt

# === Load Saved History ===
with open("training_history.pkl", "rb") as f:
    history = pickle.load(f)

acc = history["accuracy"]
loss = history["loss"]

# === Final Results ===
final_acc = acc[-1]
final_loss = loss[-1]
total_epochs = len(acc)

best_acc = max(acc)
best_acc_epoch = acc.index(best_acc) + 1

best_loss = min(loss)
best_loss_epoch = loss.index(best_loss) + 1

print(f"✅ Final Accuracy: {final_acc:.4f}")
print(f"📉 Final Loss: {final_loss:.4f}")
print(f"📊 Total Epochs Trained: {total_epochs}")
print(f"🏆 Best Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch})")
print(f"📉 Lowest Loss: {best_loss:.4f} (Epoch {best_loss_epoch})")

# === Plot Accuracy ===
plt.figure(figsize=(8, 5))
plt.plot(acc, label='Accuracy', color='blue')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.savefig("accuracy_plot.png")
plt.close()

# === Plot Loss ===
plt.figure(figsize=(8, 5))
plt.plot(loss, label='Loss', color='red')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig("loss_plot.png")
plt.close()

print("🖼️ Saved: accuracy_plot.png, loss_plot.png")

# === Save Report ===
with open("training_summary.txt", "w", encoding="utf-8") as f:
    f.write("📋 Training Summary\n")
    f.write("====================\n")
    f.write(f"✅ Final Accuracy: {final_acc:.4f}\n")
    f.write(f"📉 Final Loss: {final_loss:.4f}\n")
    f.write(f"📊 Total Epochs Trained: {total_epochs}\n")
    f.write(f"🏆 Best Accuracy: {best_acc:.4f} (Epoch {best_acc_epoch})\n")
    f.write(f"📉 Lowest Loss: {best_loss:.4f} (Epoch {best_loss_epoch})\n")
    f.write("\n🖼️ Plots saved as PNG files.\n")

print("📄 training_summary.txt generated ✅")
