using BlazorUI.Components;
using BlazorUI.Services;

var builder = WebApplication.CreateBuilder(args);

var ragServiceUrl = builder.Configuration["RagService:Url"] ?? "http://rag-service:8080";

builder.Services.AddHttpClient<RagApiClient>(client =>
{
    client.BaseAddress = new Uri(ragServiceUrl);
    client.Timeout = TimeSpan.FromMinutes(5);
});

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
}
app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);
app.UseAntiforgery();

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
