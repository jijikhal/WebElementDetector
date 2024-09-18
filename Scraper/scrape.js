result = {};
result.buttons = find_buttons();
return result;

function find_buttons() {
    found_buttons = []
    buttons = document.getElementsByTagName("a");
    for (i = 0; i < buttons.length; i++){
        element = buttons[i];
        position = element.getBoundingClientRect();
        topElt = document.elementFromPoint(position.left+position.width/2, position.top+position.height/2);
        if (topElt === null) {
            continue;
        }
        not_covered = element.isSameNode(topElt) || element.contains(topElt) || topElt.contains(element);

        if (not_covered)
            found_buttons.push(position);
    }

    return found_buttons;
}