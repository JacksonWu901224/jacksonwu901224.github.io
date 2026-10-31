// 切換側邊選單的顯示和隱藏
function toggleMenu(event) {
    event.stopPropagation();

    const sidebar = document.getElementById("sidebar");
    const main = document.getElementById("main-content");

    sidebar.classList.toggle("open");
    main.classList.toggle("shift");
}

// 當點擊頁面其他區域時，關閉側邊選單
window.addEventListener("pointerdown", function (event) {
    const sidebar = document.getElementById("sidebar");
    const menuBtn = document.querySelector(".menu-button");
    const main = document.getElementById("main-content");

    if (!sidebar.classList.contains("open")) return;

    if (sidebar.contains(event.target) || menuBtn.contains(event.target)) return;

    sidebar.classList.remove("open");
    main.classList.remove("shift");
});
// 按下 Esc 關閉側邊選單
window.addEventListener("keydown", function (event) {
    if (event.key !== "Escape") return;

    document.getElementById("sidebar").classList.remove("open");
    document.getElementById("main-content").classList.remove("shift");
});
