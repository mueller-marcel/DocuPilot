using CommunityToolkit.Mvvm.ComponentModel;

namespace DocuPilot.ViewModels;

public partial class MainWindowViewModel : ObservableObject
{
    public string Greeting { get; } = "Welcome to Avalonia!";
}
