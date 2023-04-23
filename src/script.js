// Find and hide an element in the parent frame from within the child frame, if the parent frame and the element exist, of course

const streamlitDoc = window.parent.document
if (streamlitDoc.parent) {
    const targetElement = streamlitDoc.parent.document.querySelector('.viewerBadge_container__1QSob');

    console.log(targetElement)

    if (targetElement) {
        targetElement.style.display = 'none';
    }
}