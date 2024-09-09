let number = 1;
let max_img = document.querySelectorAll('img').length;

function change_image(add) {
    old_img = document.getElementById(`img_${number}`);
    old_img.style.display = 'none';
    if (add) {
        if (max_img < number+1) {
            number = 0;
        }
        number += 1;
        new_img = document.getElementById(`img_${number}`);
        new_img.style.display = 'flex';
    } else {
        if (number == 1) {
            number = max_img+1;
        }
        number -= 1;
        new_img = document.getElementById(`img_${number}`);
        new_img.style.display = 'flex';
    }
}

function show_menu() {
    document.getElementById('navbar').style.display = 'flex';
}
