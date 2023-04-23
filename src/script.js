// Find and hide an element in the parent frame from within the child frame, if the parent frame and the element exist, of course

if (window.parent) {
    const targetElement = window.parent.document.querySelector('.viewerBadge_container__1QSob');

    if (targetElement) {
        targetElement.style.display = 'none';
    }
}